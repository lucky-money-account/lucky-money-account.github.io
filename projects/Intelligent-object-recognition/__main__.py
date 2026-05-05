import os
import sys
import logging

def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from backend.app import app
    from backend.models import preload_models

    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    print('=' * 45)
    print('  智能对象识别系统 启动中 ...')
    print('=' * 45)
    print()
    print(f'  [服务器] 后端服务启动，端口 {port}')
    preload_models()
    print('  [模型] 后台加载中，首次识别时自动可用')
    print()
    print(f'  打开浏览器访问: http://127.0.0.1:{port}')
    print('=' * 45)

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
