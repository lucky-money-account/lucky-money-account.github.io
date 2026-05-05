import os
import uuid
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from backend.config import FRONTEND_DIR
from backend.models import MODES, get_available_modes, get_model_status, preload_models

_log = logging.getLogger('objrec')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '图片过大，请上传小于 16MB 的文件'}), 413

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(get_model_status())


@app.route('/api/modes', methods=['GET'])
def api_modes():
    return jsonify({'modes': get_available_modes()})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的图片格式，请使用 jpg/png/gif/bmp/webp'}), 400

    mode = request.form.get('mode', 'general')
    if mode not in MODES:
        return jsonify({'error': f'无效的识别模式: {mode}'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        predict_fn = MODES[mode]['predict_fn']
        result = predict_fn(filepath)

        digit_extra = None
        general_extra = None

        if mode == 'digit':
            results, digit_extra = result
        elif isinstance(result, dict):
            results = result.get('results', [])
            if 'annotated_image' in result:
                general_extra = {
                    'multi_object': result.get('type') == 'multi',
                    'objects': result.get('objects', []),
                    'annotated_image': result.get('annotated_image')
                }
        else:
            results = result

        if not results:
            return jsonify({
                'mode': mode,
                'results': [],
                'message': '未识别到相关目标，请尝试其他图片或模式'
            })

        resp = {
            'mode': mode,
            'results': results,
            'top_label': result['top_label'] if isinstance(result, dict) else results[0]['label'],
            'top_confidence': result['top_confidence'] if isinstance(result, dict) else results[0]['confidence']
        }
        if digit_extra:
            resp['digit_string'] = digit_extra['string']
            resp['digit_details'] = digit_extra['details']
            if digit_extra.get('multi_row'):
                resp['multi_row'] = True
        if general_extra:
            resp.update(general_extra)
        return jsonify(resp)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        _log.exception('Prediction failed')
        return jsonify({'error': f'识别失败: {str(e)}'}), 500
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass
