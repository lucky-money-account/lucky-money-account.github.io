/* ============================================================
   MoneyForever · Blog 公共脚本
   依赖（通过 CDN 在页面中引入）：
     marked（Markdown 解析）、DOMPurify（XSS 过滤）、
     KaTeX（数学公式）、highlight.js（代码高亮，可选）
   ============================================================ */
(function (global) {
    'use strict';

    /* ── 基础工具 ── */
    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /* GitHub 风格 heading slug：小写、空白转 -、去掉标点（保留中日韩/字母/数字/-/_）
       注意：连续连字符要保留（GitHub 不折叠，例如 “GAN / Diffusion” → gan--diffusion） */
    function slugify(text) {
        return String(text || '').trim().toLowerCase()
            .replace(/\s+/g, '-')
            .replace(/[^\p{L}\p{N}\-_]/gu, '')
            .replace(/^-+|-+$/g, '');
    }

    function stripLeadingH1(md) {
        return String(md || '').replace(/^\s*#\s+[^\n]*\n+/, '');
    }

    function formatDate(iso) {
        if (!iso) return '';
        return String(iso).slice(0, 10).replace(/-/g, '.');
    }

    function readingTime(md) {
        var text = String(md || '').replace(/```[\s\S]*?```/g, ' ').replace(/\$\$[\s\S]*?\$\$/g, ' ');
        var cjk = (text.match(/[\u4e00-\u9fff]/g) || []).length;
        var words = (text.replace(/[\u4e00-\u9fff]/g, ' ').match(/[A-Za-z0-9]+/g) || []).length;
        var minutes = Math.max(1, Math.round((cjk / 350) + (words / 220)));
        return minutes + ' 分钟';
    }

    async function fetchJSON(url) {
        var sep = url.indexOf('?') === -1 ? '?' : '&';
        var res = await fetch(url + sep + 't=' + Date.now(), { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' · ' + url);
        return res.json();
    }

    /* ── 公式保护 ──
       Markdown 解析器会把公式里的 _ 当成强调、把 \ 反转义，所以先把
       $$...$$ / $...$ 换成占位符，解析完成后再用 KaTeX 渲染。
       代码块与行内代码里的 $ 不受影响。 */
    function extractMath(md) {
        var maths = [];
        var out = [];
        var lines = String(md || '').split(/\r?\n/);
        var inFence = false;
        var mathMode = false;
        var mathIndent = '';
        var mathBuf = [];

        function pushPlaceholder(tex, display) {
            maths.push({ tex: tex, display: display });
            return '@@MATH' + (display ? 'BLK' : 'INL') + (maths.length - 1) + '@@';
        }

        /* 保护一段非围栏文本里的行内代码 / $$...$$ / $...$ */
        function protectInline(text) {
            var segs = String(text).split(/(`[^`\n]*`)/g);
            for (var k = 0; k < segs.length; k++) {
                if (k % 2 === 1) continue;   /* 行内代码原样 */
                segs[k] = segs[k]
                    .replace(/\$\$([^$\n]+?)\$\$/g, function (m, tex) {
                        return pushPlaceholder(tex, true);
                    })
                    .replace(/\$([^$\n]+?)\$/g, function (m, tex) {
                        return pushPlaceholder(tex, false);
                    });
            }
            return segs.join('');
        }

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i];

            /* 代码围栏：内部原样保留 */
            if (!mathMode && /^\s*(```+|~~~+)/.test(line)) {
                inFence = !inFence;
                out.push(line);
                continue;
            }
            if (inFence) { out.push(line); continue; }

            /* 跨行 display 公式收集中 */
            if (mathMode) {
                var closeIdx = line.indexOf('$$');
                if (closeIdx === -1) { mathBuf.push(line); continue; }
                mathBuf.push(line.slice(0, closeIdx));
                out.push(mathIndent + pushPlaceholder(mathBuf.join('\n'), true));
                mathBuf = [];
                mathIndent = '';
                mathMode = false;
                var rest = line.slice(closeIdx + 2);
                if (rest.trim()) out.push(protectInline(rest));
                continue;
            }

            var trimmed = line.trim();

            /* 行首 $$：可能是同一行的 $$...$$，也可能开启跨行块 */
            if (trimmed.indexOf('$$') === 0) {
                var after = trimmed.slice(2);
                var sameLineClose = after.indexOf('$$');
                if (sameLineClose !== -1) {
                    var texSame = after.slice(0, sameLineClose);
                    var restSame = after.slice(sameLineClose + 2);
                    out.push(line.match(/^\s*/)[0] + pushPlaceholder(texSame, true) + protectInline(restSame));
                } else {
                    mathMode = true;
                    mathIndent = line.match(/^\s*/)[0];
                    if (after.trim()) mathBuf.push(after);
                }
                continue;
            }

            out.push(protectInline(line));
        }

        /* 未闭合的 $$（容错）：原样放回 */
        if (mathMode) out.push('$$' + mathBuf.join('\n'));

        return { text: out.join('\n'), maths: maths };
    }

    /* ── 把占位符替换成 KaTeX 渲染结果 ── */
    function mountMath(container, maths) {
        if (!maths.length) return;
        var re = /@@MATH(BLK|INL)(\d+)@@/g;
        var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
        var nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);

        nodes.forEach(function (node) {
            var value = node.nodeValue;
            if (!value || value.indexOf('@@MATH') === -1) return;
            re.lastIndex = 0;
            if (!re.test(value)) return;
            re.lastIndex = 0;

            var frag = document.createDocumentFragment();
            var last = 0, m;
            while ((m = re.exec(value))) {
                if (m.index > last) frag.appendChild(document.createTextNode(value.slice(last, m.index)));
                var item = maths[Number(m[2])];
                var span = document.createElement('span');
                span.className = 'math';
                if (item && typeof global.katex !== 'undefined') {
                    try {
                        global.katex.render(item.tex, span, {
                            displayMode: item.display,
                            throwOnError: false,
                            strict: 'ignore'
                        });
                    } catch (e) {
                        span.textContent = item.display ? '$$' + item.tex + '$$' : '$' + item.tex + '$';
                    }
                } else if (item) {
                    span.textContent = item.display ? '$$' + item.tex + '$$' : '$' + item.tex + '$';
                }
                frag.appendChild(span);
                last = re.lastIndex;
            }
            if (last < value.length) frag.appendChild(document.createTextNode(value.slice(last)));
            node.parentNode.replaceChild(frag, node);
        });
    }

    /* ── Markdown 渲染主入口 ──
       返回 [{id, text, level}] 目录数据 */
    function renderMarkdown(md, container) {
        if (!container) return [];
        if (typeof global.marked === 'undefined') {
            container.innerHTML =
                '<div class="error-box">渲染库（marked）加载失败，无法显示正文。请检查网络后刷新页面。</div>' +
                '<pre style="white-space:pre-wrap">' + escapeHtml(md) + '</pre>';
            return [];
        }

        var extracted = extractMath(md);
        var html;
        try {
            html = global.marked.parse(extracted.text, { gfm: true, breaks: false });
        } catch (e) {
            container.innerHTML = '<div class="error-box">Markdown 解析失败：' + escapeHtml(e.message) + '</div>';
            return [];
        }
        if (typeof global.DOMPurify !== 'undefined') {
            html = global.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
        }
        container.innerHTML = html;

        /* 外链新窗口打开 */
        container.querySelectorAll('a[href^="http"]').forEach(function (a) {
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
        });

        /* heading id + 目录：先把公式占位符还原成 $tex$，否则目录会显示 @@MATH…@@ */
        var toc = [];
        var usedIds = Object.create(null);
        container.querySelectorAll('h2, h3, h4').forEach(function (h, idx) {
            var tocText = h.textContent.replace(/@@MATH(BLK|INL)(\d+)@@/g, function (m, kind, n) {
                var item = extracted.maths[Number(n)];
                if (!item) return '';
                return item.display ? '$$' + item.tex + '$$' : '$' + item.tex + '$';
            });
            var base = slugify(h.textContent.replace(/@@MATH(?:BLK|INL)\d+@@/g, '')) || ('section-' + (idx + 1));
            var id = base, n = 1;
            while (usedIds[id]) { id = base + '-' + n; n++; }
            usedIds[id] = true;
            h.id = id;
            toc.push({ id: id, text: tocText.trim(), level: Number(h.tagName.charAt(1)) });
        });

        mountMath(container, extracted.maths);

        /* 代码高亮 */
        if (global.hljs) {
            container.querySelectorAll('pre code').forEach(function (el) {
                try { global.hljs.highlightElement(el); } catch (e) { /* ignore */ }
            });
        }

        return toc;
    }

    function renderToc(toc) {
        var lists = [document.getElementById('tocList'), document.getElementById('tocListMobile')];
        var html = (toc || []).map(function (item) {
            return '<li class="level-' + item.level + '"><a href="#' + encodeURIComponent(item.id) + '">' +
                escapeHtml(item.text) + '</a></li>';
        }).join('');
        lists.forEach(function (el) { if (el) el.innerHTML = html; });
    }

    /* 页面导航高亮：按解析后的绝对路径比较，避免 ../index.html 与 index.html 同时点亮 */
    function highlightNav() {
        var current = new URL(location.href);
        document.querySelectorAll('.site-nav a').forEach(function (a) {
            var href = a.getAttribute('href') || '';
            if (/^(https?:|#|mailto:)/.test(href)) return;
            try {
                var url = new URL(href, location.href);
                if (url.pathname === current.pathname) a.classList.add('active');
            } catch (e) { /* ignore */ }
        });
    }

    global.BlogUI = {
        escapeHtml: escapeHtml,
        slugify: slugify,
        stripLeadingH1: stripLeadingH1,
        formatDate: formatDate,
        readingTime: readingTime,
        fetchJSON: fetchJSON,
        extractMath: extractMath,
        renderMarkdown: renderMarkdown,
        renderToc: renderToc,
        highlightNav: highlightNav
    };
})(window);
