import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from flask import Flask, request, render_template, redirect, url_for, flash
import inspect
from huggingface_model_utils import load_model_by_repo_id

# Flask app
app = Flask(__name__)

# Cấu hình model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
aligner = load_model_by_repo_id('minchul/cvlface_DFA_mobilenet', os.path.expanduser('~/.cvlface_cache/minchul/cvlface_DFA_mobilenet'), os.environ['HF_TOKEN']).to(device)
fr_model = load_model_by_repo_id('minchul/cvlface_adaface_vit_base_kprpe_webface4m', os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m'), os.environ['HF_TOKEN']).to(device)

database_path = './face_db.csv'

# Hàm chuẩn hóa ảnh và căn chỉnh
def pil_to_input(pil_image, device):
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return trans(pil_image).unsqueeze(0).to(device)

def get_feat(input_tensor, aligner, fr_model, device):
    # Căn chỉnh khuôn mặt
    aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, bbox = aligner(input_tensor)

    # Nhận diện đặc trưng (feature)
    input_signature = inspect.signature(fr_model.model.net.forward)
    if input_signature.parameters.get('keypoints') is not None:
        feat = fr_model(aligned_x, aligned_ldmks)
    else:
        feat = fr_model(aligned_x)

    return feat


def compute_cosine_similarity(img1, img2, aligner, fr_model, device):
    # Chuẩn hóa ảnh
    input1 = pil_to_input(img1, device)
    input2 = pil_to_input(img2, device)

    # Lấy đặc trưng của hai ảnh
    feat1 = get_feat(input1, aligner, fr_model, device)
    feat2 = get_feat(input2, aligner, fr_model, device)

    # Tính toán cosine similarity
    cossim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    return cossim


def get_id(input_image_or_feat, database_path, aligner, fr_model, device, threshold=0.3):
    if not os.path.exists(database_path):
        return None
    db = pd.read_csv(database_path)

    # Nếu đầu vào là PIL Image, chuyển đổi thành tensor và lấy đặc trưng (feature)
    if isinstance(input_image_or_feat, Image.Image):
        input_tensor = pil_to_input(input_image_or_feat, device)
        feat_input = get_feat(input_tensor, aligner, fr_model, device)
    else:
        # Nếu đầu vào đã là tensor/feature
        feat_input = input_image_or_feat

    # So sánh với từng ảnh trong CSDL
    max_sim = -1  # Biến lưu giá trị cosine similarity lớn nhất
    matched_id = None

    for i, row in db.iterrows():
        # Chuyển đổi đặc trưng từ CSDL thành tensor
        feat_db = torch.tensor(eval(row['feat']), device=device)

        # Tính cosine similarity giữa ảnh đầu vào và ảnh trong CSDL
        cossim = torch.nn.functional.cosine_similarity(feat_input, feat_db).item()

        # Kiểm tra nếu cosine similarity lớn hơn ngưỡng và lớn hơn giá trị max_sim hiện tại
        if cossim > threshold and cossim > max_sim:
            max_sim = cossim
            matched_id = row['id']

    return matched_id

# Lưu feature mới vào CSDL
def save_to_db(feat, database_path):
    if not os.path.exists(database_path):
        db = pd.DataFrame(columns=['id', 'feat'])
        next_id = 1
    else:
        db = pd.read_csv(database_path)
        next_id = db['id'].max() + 1
    
    new_row = pd.DataFrame({'id': [next_id], 'feat': [feat.squeeze().cpu().detach().numpy().tolist()]})
    db = pd.concat([db, new_row], ignore_index=True)
    db.to_csv(database_path, index=False)
    
    return next_id

# Giao diện chính
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        pil_image = Image.open(file)
        input_tensor = pil_to_input(pil_image, device)

        # Lấy đặc trưng khuôn mặt
        feat = get_feat(input_tensor, aligner, fr_model, device)

        # Kiểm tra xem đã tồn tại trong CSDL hay chưa
        current_id = get_id(feat, database_path, aligner, fr_model, device)
        if current_id is None:
            new_id = save_to_db(feat, database_path)
            return redirect(url_for('index', message=f'Đăng ký thành công với ID: {new_id}'))
        else:
            return redirect(url_for('index', message=f'Khuôn mặt đã tồn tại với ID: {current_id}'))
    return redirect(request.url)

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        pil_image = Image.open(file)
        input_tensor = pil_to_input(pil_image, device)

        # Lấy đặc trưng khuôn mặt
        feat = get_feat(input_tensor, aligner, fr_model, device)

        # So sánh với các feature trong CSDL và tính cosine similarity
        id = get_id(feat, database_path, aligner, fr_model, device)

        if id is None:
            return redirect(url_for('index', message='Khuôn mặt chưa được đăng ký!'))
        else:
            return redirect(url_for('index', message=f'Khuôn mặt đã được nhận diện với ID: {id}'))
    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
