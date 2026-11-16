/* ============================================================
   MoneyForever · Home 2026
   中英切换 + 数据渲染；无依赖
   ============================================================ */
(function () {
    'use strict';

    /* ---------- 文案（静态部分） ---------- */
    var I18N = {
        zh: {
            'meta.title': '钱恒 · MoneyForever — 生成模型 / 计算机视觉 / 深度学习',
            'meta.desc': '钱恒（MoneyForever）的个人主页：电子科技大学软件工程，生成模型、计算机视觉与深度学习的学习笔记、项目与实验。',
            'a11y.skip': '跳到主要内容',
            'a11y.mainNav': '主导航',
            'a11y.mobileNav': '移动导航',
            'a11y.menu': '打开菜单',
            'a11y.menuOpen': '关闭菜单',
            'a11y.lang': '切换语言',
            'a11y.panel': '当前状态',
            'a11y.avatar': '阿狸头像',
            'a11y.tabs': '学习路线',
            'nav.work': '项目',
            'nav.writing': '文章',
            'nav.about': '关于',
            'nav.journey': '历程',
            'nav.blog': '博客',
            'nav.contact': '联系',
            'hero.eyebrow': 'PORTFOLIO · 2026',
            'hero.title': '你好，我是 <span class="accent">钱恒</span>，<br>也用 <span class="accent">MoneyForever</span>。',
            'hero.lede': '软件工程 @ 电子科技大学。从数学推导到能跑通的代码，喜欢把生成模型与视觉系统拆开、复现、讲清楚。',
            'hero.ctaWork': '查看项目',
            'hero.ctaWriting': '阅读笔记',
            'hero.ctaContact': '联系我',
            'hero.status': '开放 2027 暑期实习机会 · 成都',
            'hero.role': '电子科技大学 · 软件工程',
            'focus.learningLabel': '正在学',
            'focus.learning': 'Diffusion 推理 / Transformer',
            'focus.shippedLabel': '最近完成',
            'focus.shipped': '迷你 Latent Diffusion',
            'focus.focusLabel': '方向',
            'focus.focus': '生成模型 · 计算机视觉',
            'work.kicker': 'SELECTED WORK',
            'work.title': '精选项目',
            'work.sub': '从零实现的生成模型、对抗网络与视觉系统——每个项目都有可运行的代码、实验记录和结论。',
            'work.more': '更多项目',
            'writing.kicker': 'WRITING',
            'writing.title': '最近写的笔记',
            'writing.sub': '把公式推导、代码实现与踩坑记录整理成能反复读的文章。',
            'writing.all': '查看全部文章',
            'about.kicker': 'ABOUT',
            'about.title': '关于我',
            'about.p1': '电子科技大学软件工程在读。喜欢从数学推导一路写到能跑通的代码，再把过程整理成笔记——VAE、GAN、Diffusion 都是这样学过来的。',
            'about.p2': '目前的主要方向是生成模型与计算机视觉；比起调参，更在意把原理讲清楚、把实验做扎实。',
            'about.eduLabel': '教育',
            'about.edu': '电子科技大学 · 软件工程（2025 – ）',
            'about.locLabel': '所在地',
            'about.loc': '成都 · 电子科技大学沙河校区',
            'about.mailLabel': '邮箱',
            'stats.projects': '项目与实践',
            'stats.posts': '篇技术文章',
            'stats.tracks': '条学习路线',
            'journey.kicker': 'JOURNEY',
            'journey.title': '学习历程',
            'journey.sub': '三条并行的路线：夯实基础、研究 GAN、走向生成式 AI。切换标签查看每条线的节点。',
            'journey.tabBasic': '基础',
            'journey.tabGan': 'GAN',
            'journey.tabGen': '生成式 AI',
            'toolbox.kicker': 'TOOLBOX',
            'toolbox.title': '工具与技术栈',
            'toolbox.toolsTitle': '常用工具',
            'toolbox.skillsTitle': '技术栈',
            'hobbies.kicker': 'OFF-SCREEN',
            'hobbies.title': '屏幕之外',
            'contact.kicker': 'CONTACT',
            'contact.title': '聊聊？',
            'contact.sub': '项目合作、实习机会，或者深度学习上的问题，都欢迎邮件联系。',
            'contact.loc': '成都 · 中国',
            'footer.note': '用 HTML / CSS / JS 手写',
            'footer.top': '回到顶部',
            'work.count': '共 {n} 个项目',
            'blog.note': '文章以中文撰写',
            'blog.read': '阅读全文'
        },
        en: {
            'meta.title': 'Qian Heng · MoneyForever — Generative Models / Vision / Deep Learning',
            'meta.desc': 'Personal site of Qian Heng (MoneyForever): software engineering at UESTC, with notes, projects and experiments on generative models, computer vision and deep learning.',
            'a11y.skip': 'Skip to main content',
            'a11y.mainNav': 'Main navigation',
            'a11y.mobileNav': 'Mobile navigation',
            'a11y.menu': 'Open menu',
            'a11y.menuOpen': 'Close menu',
            'a11y.lang': 'Switch language',
            'a11y.panel': 'Current status',
            'a11y.avatar': 'Ali avatar',
            'a11y.tabs': 'Learning tracks',
            'nav.work': 'Work',
            'nav.writing': 'Writing',
            'nav.about': 'About',
            'nav.journey': 'Journey',
            'nav.blog': 'Blog',
            'nav.contact': 'Contact',
            'hero.eyebrow': 'PORTFOLIO · 2026',
            'hero.title': 'Hi, I\'m <span class="accent">Qian Heng</span>,<br>aka <span class="accent">MoneyForever</span>.',
            'hero.lede': 'Software engineering @ UESTC. From math to working code — I like to take generative models and vision systems apart, rebuild them, and explain them.',
            'hero.ctaWork': 'View work',
            'hero.ctaWriting': 'Read notes',
            'hero.ctaContact': 'Contact me',
            'hero.status': 'Open to summer 2027 internships · Chengdu',
            'hero.role': 'UESTC · Software Engineering',
            'focus.learningLabel': 'Learning',
            'focus.learning': 'Diffusion inference / Transformer',
            'focus.shippedLabel': 'Shipped',
            'focus.shipped': 'mini Latent Diffusion',
            'focus.focusLabel': 'Focus',
            'focus.focus': 'Generative models · CV',
            'work.kicker': 'SELECTED WORK',
            'work.title': 'Selected work',
            'work.sub': 'Generative models, adversarial nets and vision systems built from scratch — each with runnable code, experiments and conclusions.',
            'work.more': 'More projects',
            'writing.kicker': 'WRITING',
            'writing.title': 'Recent notes',
            'writing.sub': 'Derivations, implementations and debugging logs, written to be read more than once.',
            'writing.all': 'Read all articles',
            'about.kicker': 'ABOUT',
            'about.title': 'About',
            'about.p1': 'Software engineering student at UESTC. I like going from the math to running code, then turning the process into notes — that is how I learned VAE, GAN and Diffusion.',
            'about.p2': 'Currently focused on generative models and computer vision. I care more about explaining the principles and making experiments solid than about tuning numbers.',
            'about.eduLabel': 'Education',
            'about.edu': 'UESTC · Software Engineering (2025 – )',
            'about.locLabel': 'Location',
            'about.loc': 'Chengdu · UESTC Shahe Campus',
            'about.mailLabel': 'Email',
            'stats.projects': 'projects & practice',
            'stats.posts': 'technical articles',
            'stats.tracks': 'learning tracks',
            'journey.kicker': 'JOURNEY',
            'journey.title': 'Learning journey',
            'journey.sub': 'Three parallel tracks: foundations, GAN research, and generative AI. Switch tabs to see each one.',
            'journey.tabBasic': 'Foundations',
            'journey.tabGan': 'GAN',
            'journey.tabGen': 'Generative AI',
            'toolbox.kicker': 'TOOLBOX',
            'toolbox.title': 'Tools & stack',
            'toolbox.toolsTitle': 'Daily tools',
            'toolbox.skillsTitle': 'Stack',
            'hobbies.kicker': 'OFF-SCREEN',
            'hobbies.title': 'Off-screen',
            'contact.kicker': 'CONTACT',
            'contact.title': 'Say hi',
            'contact.sub': 'Whether it is a project, an internship, or a deep learning question — email is the best way to reach me.',
            'contact.loc': 'Chengdu · China',
            'footer.note': 'Hand-written with HTML / CSS / JS',
            'footer.top': 'Back to top',
            'work.count': '{n} projects',
            'blog.note': 'Articles are written in Chinese',
            'blog.read': 'Read article'
        }
    };

    /* ---------- 项目数据 ---------- */
    var PROJECTS = [
        {
            featured: true, cover: 'cover-green', index: '01',
            title: { zh: 'PBLF 图像处理课程报告', en: 'PBLF image-processing report' },
            blurb: { zh: '车牌识别（HOG + SVM）与图像分类（Iris / MNIST / 猫狗）的完整实验报告：设计、实现、测试与 21 张图表。', en: 'A full lab report on license-plate recognition (HOG + SVM) and image classification (Iris / MNIST / cats & dogs), with 21 figures.' },
            metrics: [{ v: '98%', l: { zh: 'SVM 准确率', en: 'SVM accuracy' } }, { v: '21', l: { zh: '张图表', en: 'figures' } }],
            tags: ['OpenCV', 'scikit-learn', 'SVM', 'Report'],
            links: [
                { label: { zh: '阅读报告', en: 'Read report' }, url: 'blog/post.html?p=pblf-image-processing-report' },
                { label: { zh: 'GitHub', en: 'GitHub' }, url: 'https://github.com/lucky-money-account' }
            ]
        },
        {
            featured: true, cover: 'cover-cyan', index: '02',
            title: { zh: 'Diffusion · 从零实现', en: 'Diffusion · from scratch' },
            blurb: { zh: '无条件 DDPM → 类别条件 + CFG → SRNet 4× 上采样 → 迷你 Latent Diffusion；4 个训练脚本与 2 个 notebook。', en: 'Unconditional DDPM → class conditioning + CFG → SRNet 4× upsampler → mini Latent Diffusion, with 4 training scripts and 2 notebooks.' },
            metrics: [{ v: '32→128', l: { zh: '生成分辨率', en: 'resolution' } }, { v: '4+2', l: { zh: '脚本 / notebook', en: 'scripts / notebooks' } }],
            tags: ['PyTorch', 'DDPM', 'CFG', 'Latent Diffusion'],
            links: [
                { label: { zh: '专栏', en: 'Repo' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/tree/Diffusion' },
                { label: { zh: '学习笔记', en: 'Notes' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/blob/Diffusion/diffusion.md' }
            ]
        },
        {
            featured: true, cover: 'cover-violet', index: '03',
            title: { zh: 'GAN 研究专栏', en: 'GAN research series' },
            blurb: { zh: 'DCGAN 动漫头像、DCGAN 狗图 + 自实现 FID / MiFID / KID / IS 评估、CycleGAN 黑发 ⇄ 金发图像翻译。', en: 'DCGAN anime portraits, DCGAN dogs with a self-built FID / MiFID / KID / IS suite, and CycleGAN black ⇄ blonde hair translation.' },
            metrics: [{ v: '433→91.2', l: { zh: 'FID', en: 'FID' } }, { v: '20,580', l: { zh: '狗图训练集', en: 'training images' } }],
            tags: ['DCGAN', 'CycleGAN', 'FID', 'PyTorch'],
            links: [
                { label: { zh: '母分支', en: 'Repo' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/tree/GAN-research' },
                { label: { zh: 'CycleGAN', en: 'CycleGAN' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/tree/GAN-CycleGAN' }
            ]
        },
        {
            featured: false, cover: 'cover-amber', index: '04',
            title: { zh: '动手学深度学习 · 笔记与实践', en: 'Dive into Deep Learning · notes & practice' },
            blurb: { zh: '《动手学深度学习》(PyTorch) 的章节知识点、学习心得、综合题与练习 Notebook，覆盖 ch02–ch06 与 GAN 专题。', en: 'Chapter notes, reflections, exercises and notebooks for Dive into Deep Learning (PyTorch): ch02–ch06 plus a GAN专题.' },
            tags: ['PyTorch', 'Jupyter', 'CNN', 'GAN'],
            links: [{ label: { zh: '仓库', en: 'Repo' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning' }]
        },
        {
            featured: false, cover: 'cover-cyan', index: '05',
            title: { zh: '图像变换与增强', en: 'Image transforms & enhancement' },
            blurb: { zh: 'OpenCV 视频逐帧灰度化、缩放、高斯去噪、直方图均衡化，并做单帧效果展示。', en: 'Frame-by-frame grayscale, resize, Gaussian denoising and histogram equalisation on video, with single-frame comparisons.' },
            tags: ['Python', 'OpenCV'],
            links: [{ label: { zh: '作业', en: 'Notebook' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/blob/PBLF-Image-Processing/video_processing_modified.ipynb' }]
        },
        {
            featured: false, cover: 'cover-violet', index: '06',
            title: { zh: 'kNN 手写数字识别', en: 'kNN digit recognition' },
            blurb: { zh: 'OpenCV digits（92.56%）与 MNIST（97.71%）两套数据，含多种预处理对照实验。', en: 'OpenCV digits (92.56%) and MNIST (97.71%), with several preprocessing comparisons.' },
            tags: ['scikit-learn', 'kNN', 'MNIST'],
            links: [{ label: { zh: '笔记', en: 'Notes' }, url: 'https://github.com/lucky-money-account/repository_for_machine_learning/blob/kNN/learning_notes.md' }]
        },
        {
            featured: false, cover: 'cover-cyan', index: '07',
            title: { zh: 'Weather Forcast · 西南天气预测', en: 'Weather forecast · Southwest China' },
            blurb: { zh: '基于 LSTM / Transformer 与真实 ERA5 气象数据的预测系统，配 Streamlit 交互面板。', en: 'Forecasting system built on LSTM / Transformer and real ERA5 data, with a Streamlit dashboard.' },
            tags: ['LSTM', 'Transformer', 'Streamlit'],
            links: [{ label: { zh: '项目', en: 'Repo' }, url: 'https://github.com/lucky-money-account/Weather-Forcast' }]
        },
        {
            featured: false, cover: 'cover-green', index: '08',
            title: { zh: '智能对象识别系统', en: 'Object recognition system' },
            blurb: { zh: 'YOLOv8 + MobileNetV2 + Flask 的多模式识别：通用物体、数字、动物、场景。', en: 'Multi-mode recognition with YOLOv8 + MobileNetV2 + Flask: objects, digits, animals and scenes.' },
            tags: ['YOLOv8', 'MobileNetV2', 'Flask'],
            links: [{ label: { zh: '项目', en: 'Repo' }, url: 'https://github.com/lucky-money-account/Intelligent-object-recognition' }]
        },
        {
            featured: false, cover: 'cover-amber', index: '09',
            title: { zh: 'PBLF 猫狗图像分类实验', en: 'PBLF cats vs dogs experiment' },
            blurb: { zh: '浅层 CNN → 手写 VGG16 → 预训练 VGG16 迁移学习，4 个延伸实验。', en: 'Shallow CNN → hand-written VGG16 → pretrained VGG16 transfer learning, with four extension experiments.' },
            tags: ['VGG16', 'ResNet', 'Transfer Learning'],
            links: [{ label: { zh: '项目', en: 'Repo' }, url: 'https://github.com/lucky-money-account/PBLF-catdog-classification-experiment' }]
        },
        {
            featured: false, cover: 'cover-cyan', index: '10',
            title: { zh: 'DL 函数查询助手', en: 'DL function helper' },
            blurb: { zh: '本地化深度学习函数速查工具：196 个函数、语义搜索、单文件 exe 双击即用。', en: 'Offline deep-learning function lookup: 196 functions, semantic search, shipped as a single executable.' },
            tags: ['tkinter', 'TF-IDF', 'PyInstaller'],
            links: [{ label: { zh: '项目', en: 'Repo' }, url: 'https://github.com/lucky-money-account/Deep-Learning-Function-Helper' }]
        }
    ];

    /* ---------- 时间线 ---------- */
    var JOURNEY = {
        basic: [
            { date: '2025.09', title: { zh: '考入电子科技大学软件学院', en: 'Admitted to UESTC Software Engineering' }, desc: { zh: '开始编程之旅，从零起步探索计算机世界。', en: 'The journey started — exploring computing from zero.' } },
            { date: '2025.11', title: { zh: '开始接触机器学习', en: 'First steps in machine learning' }, desc: { zh: '入门 ML 基础概念，了解监督学习与非监督学习。', en: 'Learned the basics of supervised and unsupervised learning.' } },
            { date: '2025.12', title: { zh: '开始接触计算机视觉', en: 'First steps in computer vision' }, desc: { zh: '图像处理、OpenCV 以及视觉任务的初步实践。', en: 'Image processing, OpenCV and first vision exercises.' } },
            { date: '2026.03', title: { zh: '系统学习深度学习（PBLF）', en: 'Systematic deep learning (PBLF)' }, desc: { zh: '以项目驱动方式深入学习 PyTorch、神经网络与模型训练。', en: 'Project-driven study of PyTorch, neural networks and training.' } },
            { date: '2026.05', title: { zh: '完成多个实践项目', en: 'Shipped several practice projects' }, desc: { zh: '图像分类、西南天气预测、智能对象识别。', en: 'Image classification, weather forecasting and object recognition.' } },
            { date: '2026.08', title: { zh: '重头系统学习深度学习基础', en: 'Rebuilding the foundations' }, desc: { zh: '回归基础，系统梳理深度学习理论，夯实根基。', en: 'Back to fundamentals — rebuilding the theory from the ground up.' } }
        ],
        gan: [
            { date: '2026.09', title: { zh: '开启 GAN 研究专栏', en: 'Started the GAN research series' }, desc: { zh: '从 DCGAN 起步，建立「母分支 + 一项目一分支」的研究体系。', en: 'Starting from DCGAN with a "mother branch + one branch per project" setup.' } },
            { date: '2026.09', title: { zh: 'DCGAN · 动漫头像生成', en: 'DCGAN · anime portraits' }, desc: { zh: '用 Kaggle 动漫人脸（63,565 张）从 100 维噪声生成 64×64 头像。', en: 'Generated 64×64 anime portraits from 100-dim noise with 63,565 Kaggle faces.' } },
            { date: '2026.09', title: { zh: 'DCGAN · 狗图生成 + FID', en: 'DCGAN · dogs + FID' }, desc: { zh: '自建 FID / MiFID / KID / IS 评估体系，FID 从 433 降到 91.2。', en: 'Built a FID / MiFID / KID / IS evaluation suite and drove FID from 433 to 91.2.' } },
            { date: '2026.09', title: { zh: 'CycleGAN · 黑发 ⇄ 金发', en: 'CycleGAN · black ⇄ blonde hair' }, desc: { zh: '非配对图像翻译：CelebA 按发色拆成两个域，用循环一致约束学习双向映射。', en: 'Unpaired translation: CelebA split by hair colour, learned with cycle-consistency.' } },
            { date: { zh: '进行中', en: 'Ongoing' }, current: true, title: { zh: '路线图 · WGAN / CGAN / 潜空间探索', en: 'Roadmap · WGAN / CGAN / latent space' }, desc: { zh: '下一步：WGAN-GP 改善稳定性、条件生成、潜空间插值。', en: 'Next: WGAN-GP stability, conditional generation and latent-space interpolation.' } }
        ],
        gen: [
            { date: '2026.09', title: { zh: 'VAE · 变分自编码器', en: 'VAE · variational autoencoder' }, desc: { zh: '从零推导 ELBO / 重参数化 / 高斯 KL 闭式，在眼镜人脸数据集上训练并做潜空间属性编辑。', en: 'Derived ELBO / reparameterisation / closed-form Gaussian KL, trained on a glasses dataset and edited attributes in latent space.' } },
            { date: '2026.10', title: { zh: '分层 VAE（HVAE）', en: 'Hierarchical VAE (HVAE)' }, desc: { zh: '把 ELBO 推广到多层潜变量，理解它作为扩散模型数学桥的角色。', en: 'Extended ELBO to multiple latent layers — the mathematical bridge to diffusion.' } },
            { date: '2026.11', current: true, title: { zh: 'Diffusion · DDPM → CFG → Latent', en: 'Diffusion · DDPM → CFG → Latent' }, desc: { zh: '从零实现无条件 DDPM，扩展类别条件与 CFG，接入 SRNet 4× 上采样，完成迷你 Latent Diffusion。', en: 'Built unconditional DDPM from scratch, added CFG and a 4× SR upsampler, then a mini Latent Diffusion.' } },
            { date: { zh: '规划中', en: 'Planned' }, title: { zh: 'VQ-VAE · 离散码本', en: 'VQ-VAE · discrete codebook' }, desc: { zh: '最近邻量化 + 直通估计 + commitment loss，进入离散潜空间。', en: 'Nearest-neighbour quantisation, straight-through estimation and commitment loss.' } },
            { date: { zh: '规划中', en: 'Planned' }, title: { zh: 'Transformer · 从零手写', en: 'Transformer · from scratch' }, desc: { zh: '自注意力 / 多头 / 位置编码，为 VQ-GAN 的自回归先验打底。', en: 'Self-attention, multi-head and positional encoding for a VQ-GAN prior.' } },
            { date: { zh: '规划中', en: 'Planned' }, title: { zh: 'VQ-GAN', en: 'VQ-GAN' }, desc: { zh: 'VQ-VAE + 对抗 + 感知损失 + Transformer 先验，做高保真生成。', en: 'VQ-VAE + adversarial + perceptual losses + a Transformer prior.' } }
        ]
    };

    var TOOLS = [
        { name: 'VSCode', desc: { zh: '代码编辑器，主力开发环境。', en: 'Code editor — my main workspace.' } },
        { name: 'Anaconda', desc: { zh: 'Python 环境与包管理。', en: 'Python environments and packages.' } },
        { name: 'Jupyter', desc: { zh: '实验、可视化与学习笔记。', en: 'Experiments, plots and notes.' } },
        { name: 'Git & GitHub', desc: { zh: '版本管理与代码托管。', en: 'Version control and hosting.' } }
    ];

    var SKILLS = [
        { label: { zh: '语言', en: 'Languages' }, items: ['Python', 'C / C++', 'JavaScript', 'SQL'] },
        { label: { zh: '框架 / 库', en: 'Frameworks' }, items: ['PyTorch', 'TensorFlow / Keras', 'OpenCV', 'scikit-learn', 'Flask', 'Streamlit'] },
        { label: { zh: '工具', en: 'Tools' }, items: ['Git', 'Linux', 'Anaconda', 'LaTeX'] },
        { label: { zh: '正在学', en: 'Learning' }, items: ['Diffusion', 'Transformer', 'VQ-VAE'] }
    ];

    var HOBBIES = [
        { icon: '💻', name: { zh: '编程', en: 'Coding' }, desc: { zh: '折腾新工具，也喜欢把想法写成能跑的代码。', en: 'Tinkering with new tools and turning ideas into code.' } },
        { icon: '📚', name: { zh: '阅读', en: 'Reading' }, desc: { zh: '技术书、科幻与历史。', en: 'Tech books, sci-fi and history.' } },
        { icon: '🥾', name: { zh: '徒步', en: 'Hiking' }, desc: { zh: '去山里走走，看看远一点的风景。', en: 'Getting outside and going a bit further.' } },
        { icon: '🎮', name: { zh: 'Minecraft', en: 'Minecraft' }, desc: { zh: '欢迎找我联机。', en: 'Always up for a co-op session.' } }
    ];

    /* ---------- 工具函数 ---------- */
    var $ = function (sel, root) { return (root || document).querySelector(sel); };
    var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function L(value) {
        if (value && typeof value === 'object' && (value.zh || value.en)) return value[lang] || value.zh;
        return value;
    }

    function t(key, vars) {
        var text = (I18N[lang] && I18N[lang][key]) || (I18N.zh[key] || key);
        if (vars) Object.keys(vars).forEach(function (k) { text = text.replace('{' + k + '}', vars[k]); });
        return text;
    }

    function svgArrow() {
        return '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M3 8h8.6L8.3 4.7l1.1-1.1L14 8l-4.6 4.4-1.1-1.1L11.6 8H3V8Z"/></svg>';
    }
    function svgExternal() {
        return '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M9 2h5v5h-1.5V4.6L7.1 10 6 8.9l5.4-5.4H9V2ZM3 4h4v1.5H4.5v6h6V9H12v4H3V4Z"/></svg>';
    }

    /* ---------- 语言 ---------- */
    var lang = 'zh';
    try {
        var urlLang = new URLSearchParams(window.location.search).get('lang');
        var saved = localStorage.getItem('mf-lang');
        if (urlLang === 'en' || urlLang === 'zh') lang = urlLang;
        else if (saved === 'en' || saved === 'zh') lang = saved;
    } catch (e) { /* ignore */ }

    function applyI18n() {
        document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
        document.title = t('meta.title');
        var desc = $('meta[name="description"]');
        if (desc) desc.setAttribute('content', t('meta.desc'));

        $$('[data-i18n]').forEach(function (el) { el.textContent = t(el.getAttribute('data-i18n')); });
        $$('[data-i18n-html]').forEach(function (el) { el.innerHTML = t(el.getAttribute('data-i18n-html')); });
        $$('[data-i18n-aria]').forEach(function (el) { el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria'))); });

        var toggle = $('#langToggle');
        if (toggle) toggle.textContent = lang === 'zh' ? 'EN' : '中文';

        var menuBtn = $('#menuBtn');
        if (menuBtn) {
            var open = menuBtn.getAttribute('aria-expanded') === 'true';
            menuBtn.setAttribute('aria-label', t(open ? 'a11y.menuOpen' : 'a11y.menu'));
        }
    }

    function setLang(next) {
        lang = next === 'en' ? 'en' : 'zh';
        try { localStorage.setItem('mf-lang', lang); } catch (e) { /* ignore */ }
        applyI18n();
        renderProjects();
        renderTimeline();
        renderTools();
        renderSkills();
        renderHobbies();
        renderWriting(postsCache);
    }

    /* ---------- 渲染：项目 ---------- */
    function renderProjects() {
        var featuredBox = $('#featuredProjects');
        var listBox = $('#projectList');
        if (!featuredBox || !listBox) return;

        var featured = PROJECTS.filter(function (p) { return p.featured; });
        var others = PROJECTS.filter(function (p) { return !p.featured; });

        featuredBox.innerHTML = featured.map(function (p) {
            var metrics = p.metrics.map(function (m) {
                return '<span>' + escapeHtml(m.v) + ' · ' + escapeHtml(L(m.l)) + '</span>';
            }).join('');
            var tags = p.tags.map(function (tag) { return '<span class="tag">' + escapeHtml(tag) + '</span>'; }).join('');
            var links = p.links.map(function (link) {
                return '<a class="link-arrow" href="' + escapeHtml(link.url) + '"' +
                    (link.url.indexOf('http') === 0 ? ' target="_blank" rel="noopener"' : '') + '>' +
                    escapeHtml(L(link.label)) + svgExternal() + '</a>';
            }).join('');
            return '<article class="feature-card">' +
                '<div class="cover ' + p.cover + '">' +
                    '<span class="cover-index">' + p.index + ' / ' + escapeHtml(L(p.title).split('·')[0].trim()) + '</span>' +
                    '<div class="cover-metrics">' + metrics + '</div>' +
                '</div>' +
                '<div class="feature-body">' +
                    '<h3>' + escapeHtml(L(p.title)) + '</h3>' +
                    '<p>' + escapeHtml(L(p.blurb)) + '</p>' +
                    '<div class="tags">' + tags + '</div>' +
                    '<div class="feature-links">' + links + '</div>' +
                '</div>' +
            '</article>';
        }).join('');

        listBox.innerHTML = others.map(function (p) {
            var tags = p.tags.map(function (tag) { return '<span class="tag">' + escapeHtml(tag) + '</span>'; }).join('');
            var primary = p.links[0] || { url: '#', label: p.title };
            return '<a class="project-row" href="' + escapeHtml(primary.url) + '"' +
                (primary.url.indexOf('http') === 0 ? ' target="_blank" rel="noopener"' : '') + '>' +
                '<span class="row-index">' + p.index + '</span>' +
                '<span class="row-body">' +
                    '<h4>' + escapeHtml(L(p.title)) + '</h4>' +
                    '<p>' + escapeHtml(L(p.blurb)) + '</p>' +
                    '<span class="row-tags">' + tags + '</span>' +
                '</span>' +
                '<span class="row-arrow" aria-hidden="true">' + svgExternal() + '</span>' +
            '</a>';
        }).join('');

        var countEl = $('#projectCount');
        if (countEl) countEl.textContent = t('work.count', { n: PROJECTS.length });
    }

    /* ---------- 渲染：时间线 ---------- */
    var activeTrack = 'basic';

    function renderTimeline() {
        var box = $('#timelineList');
        if (!box) return;
        var entries = JOURNEY[activeTrack] || [];
        box.innerHTML = entries.map(function (item) {
            return '<li class="tl-item' + (item.current ? ' is-current' : '') + '">' +
                '<span class="tl-date">' + escapeHtml(L(item.date)) + '</span>' +
                '<div class="tl-body">' +
                    '<div class="tl-title">' + escapeHtml(L(item.title)) + '</div>' +
                    '<div class="tl-desc">' + escapeHtml(L(item.desc)) + '</div>' +
                '</div>' +
            '</li>';
        }).join('');
    }

    function initTabs() {
        var tabs = $$('.tab');
        tabs.forEach(function (tab) {
            tab.addEventListener('click', function () {
                activeTrack = tab.getAttribute('data-track') || 'basic';
                tabs.forEach(function (x) {
                    var on = x === tab;
                    x.classList.toggle('is-active', on);
                    x.setAttribute('aria-selected', on ? 'true' : 'false');
                });
                renderTimeline();
            });
        });
    }

    /* ---------- 渲染：工具 / 技能 / 爱好 ---------- */
    function renderTools() {
        var box = $('#toolList');
        if (!box) return;
        box.innerHTML = TOOLS.map(function (tool) {
            return '<li class="tool-item"><span class="tool-name">' + escapeHtml(tool.name) + '</span>' +
                '<span class="tool-desc">' + escapeHtml(L(tool.desc)) + '</span></li>';
        }).join('');
    }

    function renderSkills() {
        var box = $('#skillGroups');
        if (!box) return;
        box.innerHTML = SKILLS.map(function (group) {
            var chips = group.items.map(function (item) { return '<span class="chip">' + escapeHtml(item) + '</span>'; }).join('');
            return '<div class="skill-group"><div class="skill-label">' + escapeHtml(L(group.label)) + '</div>' +
                '<div class="chips">' + chips + '</div></div>';
        }).join('');
    }

    function renderHobbies() {
        var box = $('#hobbyGrid');
        if (!box) return;
        box.innerHTML = HOBBIES.map(function (hobby) {
            return '<div class="hobby-card"><div class="hobby-icon" aria-hidden="true">' + hobby.icon + '</div>' +
                '<div class="hobby-name">' + escapeHtml(L(hobby.name)) + '</div>' +
                '<div class="hobby-desc">' + escapeHtml(L(hobby.desc)) + '</div></div>';
        }).join('');
    }

    /* ---------- 渲染：文章 ---------- */
    var postsCache = null;

    function formatDate(iso) {
        if (!iso) return '';
        var parts = String(iso).split('-');
        if (parts.length < 3) return iso;
        return lang === 'zh' ? parts[0] + '.' + parts[1] + '.' + parts[2] : parts[0] + '-' + parts[1] + '-' + parts[2];
    }

    function renderWriting(posts) {
        var box = $('#writingList');
        if (!box) return;
        var list = (posts || []).slice(0, 3);
        if (!list.length) {
            box.innerHTML = '<div class="blog-fallback"><a class="link-arrow" href="blog/index.html">' +
                escapeHtml(t('writing.all')) + svgArrow() + '</a></div>';
            return;
        }
        box.innerHTML = list.map(function (post) {
            var tags = (post.tags || []).slice(0, 2).map(function (tag) {
                return '<span class="tag">' + escapeHtml(tag) + '</span>';
            }).join('');
            var summary = post.summary || '';
            return '<a class="blog-card" href="blog/post.html?p=' + encodeURIComponent(post.slug) + '">' +
                '<span class="blog-top">' +
                    '<span class="blog-date">' + escapeHtml(formatDate(post.date)) + '</span>' +
                    '<span class="blog-tags">' + tags + '</span>' +
                '</span>' +
                '<span class="blog-title">' + escapeHtml(post.title || post.slug) + '</span>' +
                '<span class="blog-summary">' + escapeHtml(summary) + '</span>' +
                (lang === 'en' ? '<span class="blog-note">' + escapeHtml(t('blog.note')) + '</span>' : '') +
                '<span class="blog-read">' + escapeHtml(t('blog.read')) + svgArrow() + '</span>' +
            '</a>';
        }).join('');
    }

    function loadWriting() {
        if (!window.fetch) return;
        fetch('blog/posts.json', { cache: 'no-store' })
            .then(function (res) { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
            .then(function (data) {
                postsCache = (data.posts || []).slice().sort(function (a, b) {
                    return String(b.date || '').localeCompare(String(a.date || ''));
                });
                renderWriting(postsCache);
            })
            .catch(function () { renderWriting(null); });
    }

    /* ---------- 交互：菜单 / 语言 / 导航高亮 / 动效 ---------- */
    function initMenu() {
        var btn = $('#menuBtn');
        var nav = $('#mobileNav');
        if (!btn || !nav) return;

        function close() {
            nav.hidden = true;
            btn.setAttribute('aria-expanded', 'false');
            btn.setAttribute('aria-label', t('a11y.menu'));
        }
        function open() {
            nav.hidden = false;
            btn.setAttribute('aria-expanded', 'true');
            btn.setAttribute('aria-label', t('a11y.menuOpen'));
        }

        btn.addEventListener('click', function () {
            if (nav.hidden) open(); else close();
        });
        $$('a', nav).forEach(function (link) { link.addEventListener('click', close); });
        document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
        window.addEventListener('resize', function () { if (window.innerWidth > 760) close(); });
    }

    function initLang() {
        var btn = $('#langToggle');
        if (!btn) return;
        btn.addEventListener('click', function () { setLang(lang === 'zh' ? 'en' : 'zh'); });
    }

    function initNavHighlight() {
        if (!('IntersectionObserver' in window)) return;
        var sections = $$('main section[id]');
        if (!sections.length) return;
        var links = $$('.desktop-nav a');
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                var id = entry.target.getAttribute('id');
                links.forEach(function (link) {
                    link.classList.toggle('is-active', link.getAttribute('href') === '#' + id);
                });
            });
        }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
        sections.forEach(function (section) { observer.observe(section); });
    }

    function initReveal() {
        var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        var targets = $$('.section-head, .feature-card, .project-row, .blog-card, .stat, .panel, .hobby-card, .contact-card');
        if (reduce || !('IntersectionObserver' in window)) return;
        targets.forEach(function (el) { el.classList.add('reveal'); });
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -32px 0px' });
        targets.forEach(function (el) { observer.observe(el); });
    }

    /* ---------- 初始化 ---------- */
    function init() {
        applyI18n();
        initMenu();
        initLang();
        initTabs();
        renderProjects();
        renderTimeline();
        renderTools();
        renderSkills();
        renderHobbies();
        loadWriting();
        initNavHighlight();
        initReveal();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
