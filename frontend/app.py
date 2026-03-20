from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

AI_SERVICE_URL   = os.getenv('AI_SERVICE_URL',   'http://ai-service:5001')
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://data-service:5002')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    ai_resp = requests.post(f'{AI_SERVICE_URL}/classify', json=data, timeout=30)
    ai_resp.raise_for_status()
    result = ai_resp.json()

    save_payload = {
        'filename':        data.get('filename', 'image.jpg'),
        'image_base64':    data.get('image', ''),
        'top_label':       result.get('species', ''),
        'common_name':     result.get('common_name', ''),
        'top_confidence':  result.get('confidence', 0) / 100,
        'danger':          result.get('danger', 'NO_WILDLIFE'),
        'venomous':        result.get('venomous', False),
        'aggressive':      result.get('aggressive', False),
        'action':          result.get('action', ''),
        'is_wildlife':     result.get('is_wildlife', False),
        'all_predictions': result.get('predictions', []),
    }
    requests.post(f'{DATA_SERVICE_URL}/save', json=save_payload, timeout=10)
    return jsonify(result)


@app.route('/api/history')
def history():
    page  = request.args.get('page', 1)
    limit = request.args.get('limit', 12)
    resp  = requests.get(f'{DATA_SERVICE_URL}/history?page={page}&limit={limit}', timeout=10)
    return jsonify(resp.json())


@app.route('/api/stats')
def stats():
    resp = requests.get(f'{DATA_SERVICE_URL}/stats', timeout=10)
    return jsonify(resp.json())


@app.route('/api/delete/<int:record_id>', methods=['DELETE'])
def delete(record_id):
    resp = requests.delete(f'{DATA_SERVICE_URL}/delete/{record_id}', timeout=10)
    return jsonify(resp.json())


@app.route('/api/reclassify/<int:record_id>', methods=['POST'])
def reclassify(record_id):
    hist_resp = requests.get(f'{DATA_SERVICE_URL}/history?page=1&limit=1000', timeout=10)
    items = hist_resp.json().get('items', [])
    item  = next((i for i in items if i['id'] == record_id), None)
    if not item:
        return jsonify({'error': 'Record not found'}), 404
    ai_resp = requests.post(
        f'{AI_SERVICE_URL}/classify',
        json={'image': item['image_base64'], 'filename': item['filename']},
        timeout=30,
    )
    return jsonify(ai_resp.json())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
