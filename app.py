import os
import cv2
import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from cellpose.models import Cellpose
import base64
import io
import zipfile
import uuid
from threading import Thread
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB feltöltési limit

tasks = {}

# --- ÁBRA KÉSZÍTŐ FÜGGVÉNYEK ---

def create_histogram(intensity_data, channel_order):
    plt.style.use('default')
    num_channels = len(channel_order)
    if num_channels == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
    elif num_channels == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 4), sharey=True)
    
    if num_channels > 0:
        colors = ['#007bff', '#28a745', '#dc3545', '#ffc107']
        for i, channel in enumerate(channel_order):
            if intensity_data.get(channel):
                axes[i].hist(intensity_data[channel], bins=25, color=colors[i % len(colors)], alpha=0.8)
            axes[i].set_title(channel, fontsize=10)
            axes[i].set_xlabel('Mean Intensity', fontsize=8)
            axes[i].tick_params(axis='x', labelsize=7); axes[i].tick_params(axis='y', labelsize=7)
        axes[0].set_ylabel('Cell Count', fontsize=8)
        fig.suptitle('Intensity Distribution per Channel', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    img_bytes = io.BytesIO(); fig.savefig(img_bytes, format='png', dpi=300); plt.close(fig); img_bytes.seek(0)
    return img_bytes

def create_boxplot(intensity_data, channel_order):
    plt.style.use('default')
    normalized_data, valid_channels = [], []
    for channel in channel_order:
        if intensity_data.get(channel):
            channel_data = intensity_data[channel]
            if channel_data: # Check if list is not empty
                max_val = max(channel_data)
                if max_val > 0:
                    normalized_data.append([val / max_val * 100 for val in channel_data])
                    valid_channels.append(channel)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if not normalized_data:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
    else:
        box = ax.boxplot(normalized_data, labels=valid_channels, patch_artist=True, widths=0.6)
        colors = ['#007bff', '#28a745', '#dc3545', '#ffc107']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title('Normalized Intensity Distribution per Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Max Intensity', fontsize=9); ax.set_xlabel('Fluorescent Dye Type', fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
    
    fig.tight_layout()
    img_bytes = io.BytesIO(); fig.savefig(img_bytes, format='png', dpi=300); plt.close(fig); img_bytes.seek(0)
    return img_bytes

def create_area_histogram(df):
    plt.style.use('default'); fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty and 'Area_pixels' in df:
        ax.hist(df['Area_pixels'], bins=30, color='#6c757d', alpha=0.8)
    ax.set_title('Cell Area Distribution', fontsize=12, fontweight='bold'); ax.set_xlabel('Area (pixels²)', fontsize=9); ax.set_ylabel('Cell Count', fontsize=9)
    ax.tick_params(axis='both', labelsize=8); fig.tight_layout(); img_bytes = io.BytesIO(); fig.savefig(img_bytes, format='png', dpi=300); plt.close(fig); img_bytes.seek(0)
    return img_bytes

def create_cell_count_plot(initial_count, final_count):
    plt.style.use('default'); fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Initial', 'Filtered']; counts = [initial_count, final_count]
    bars = ax.bar(labels, counts, color=['#007bff', '#28a745']); ax.set_ylabel('Cell Count', fontsize=9)
    ax.set_title('Cell Count Before & After Filtering', fontsize=12, fontweight='bold'); ax.bar_label(bars, padding=3, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    if counts and max(counts) > 0: ax.set_ylim(top=max(counts) * 1.15)
    fig.tight_layout(); img_bytes = io.BytesIO(); fig.savefig(img_bytes, format='png', dpi=300); plt.close(fig); img_bytes.seek(0)
    return img_bytes

def create_channel_comparison_plot(results_dict):
    plt.style.use('default'); fig, ax = plt.subplots(figsize=(8, 5))
    channels = list(results_dict.keys())
    counts = [res['count'] for res in results_dict.values()]
    bars = ax.bar(channels, counts, color=['#007bff', '#28a745', '#dc3545', '#ffc107'])
    ax.set_ylabel('Final Cell Count', fontsize=9)
    ax.set_title('Comparative Cell Count per Channel', fontsize=12, fontweight='bold')
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    if counts and max(counts) > 0: ax.set_ylim(top=max(counts) * 1.15)
    fig.tight_layout(); img_bytes = io.BytesIO(); fig.savefig(img_bytes, format='png', dpi=300); plt.close(fig); img_bytes.seek(0)
    return img_bytes

def reconstruct_image(channels, channel_name, final_cell_df, masks_dict, tile_metadata):
    if not channel_name in channels or not channels[channel_name]: return None
    
    max_x = max(t['x'] for t in channels[channel_name])
    max_y = max(t['y'] for t in channels[channel_name])
    full_width = max_x + tile_metadata['tile_size']
    full_height = max_y + tile_metadata['tile_size']

    reconstructed_image = np.zeros((full_height, full_width), dtype=np.uint8)
    for tile in channels[channel_name]:
        img_tile = cv2.imdecode(np.frombuffer(tile['data'], np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = img_tile.shape
        reconstructed_image[tile['y']:tile['y']+h, tile['x']:tile['x']+w] = img_tile
    
    overlay_img = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)
    if not final_cell_df.empty:
        for row in final_cell_df.itertuples():
            cell_id, global_cx, global_cy = row.Cell_ID, row.Global_CX, row.Global_CY
            
            for tile_coord, mask_data in masks_dict.items():
                if cell_id in mask_data['cell_map']:
                    original_id = mask_data['cell_map'][cell_id]
                    mask_on_tile = (mask_data['mask'] == original_id)
                    
                    full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                    tile_y, tile_x = tile_coord
                    h, w = mask_on_tile.shape
                    full_mask[tile_y:tile_y+h, tile_x:tile_x+w] = mask_on_tile
                    
                    np.random.seed(cell_id)
                    color = np.random.randint(50, 255, size=3).tolist()
                    overlay_img[full_mask.astype(bool)] = color
                    cv2.putText(overlay_img, str(cell_id), (global_cx - 10, global_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                    break
    
    _, buffer = cv2.imencode('.png', overlay_img)
    return buffer.tobytes()

def run_quantification_mode(task_id, channels, tile_metadata, thresholds, bg_coords, comp_matrix):
    seg_channel_name = thresholds['seg_channel']
    if seg_channel_name not in channels: raise ValueError(f"Segmentation channel '{seg_channel_name}' not found.")
    
    all_cells_data, model, masks_dict = [], Cellpose(model_type='cyto', gpu=True), {}
    total_tiles = len(channels[seg_channel_name])
    
    for i, tile in enumerate(sorted(channels[seg_channel_name], key=lambda t: (t['y'], t['x']))):
        tasks[task_id]['status'] = f'Analyzing tile {i+1}/{total_tiles}...'; tasks[task_id]['progress'] = 10 + int(70 * (i / total_tiles))
        
        seg_image_raw = cv2.imdecode(np.frombuffer(tile['data'], np.uint8), cv2.IMREAD_GRAYSCALE)
        masks, _, _, _ = model.eval(seg_image_raw, diameter=None, channels=[0,0])
        
        current_cell_map = {}
        for cell_id_on_tile in np.unique(masks)[1:]:
            cell_mask = (masks == cell_id_on_tile)
            M = cv2.moments(cell_mask.astype(np.uint8))
            if M["m00"] > 0:
                local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                o = tile_metadata.get('overlap', 0)
                if not (tile['x'] > 0 and local_cx < o) and not (tile['y'] > 0 and local_cy < o):
                    new_cell_id = len(all_cells_data) + 1
                    cell_info = {'Cell_ID': new_cell_id, 'Global_CX': tile['x'] + local_cx, 'Global_CY': tile['y'] + local_cy, 'Area_pixels': int(np.sum(cell_mask))}
                    for ch_name, ch_tiles in channels.items():
                        corr_tile_data = next((t['data'] for t in ch_tiles if t['x'] == tile['x'] and t['y'] == tile['y']), None)
                        if corr_tile_data:
                            ch_img = cv2.imdecode(np.frombuffer(corr_tile_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                            cell_info[f'Mean_Intensity_{ch_name}'] = float(np.mean(ch_img[cell_mask]))
                    current_cell_map[new_cell_id] = cell_id_on_tile
                    all_cells_data.append(cell_info)
        
        masks_dict[(tile['y'], tile['x'])] = {'mask': masks, 'cell_map': current_cell_map}

    df = pd.DataFrame(all_cells_data) if all_cells_data else pd.DataFrame()
    initial_cell_count = len(df)

    tasks[task_id]['status'] = 'Filtering results...'; tasks[task_id]['progress'] = 85
    df_filtered = df.copy()
    if not df.empty:
        df_filtered = df[(df['Area_pixels'] >= thresholds['min_area']) & (df['Area_pixels'] <= thresholds['max_area'])]
        intensity_col = f'Mean_Intensity_{seg_channel_name}'
        if intensity_col in df_filtered:
            df_filtered = df_filtered[(df_filtered[intensity_col] >= thresholds['min_intensity']) & (df_filtered[intensity_col] <= thresholds['max_intensity'])]
    final_cell_count = len(df_filtered)

    tasks[task_id]['status'] = 'Generating results...'; tasks[task_id]['progress'] = 90
    channel_order = list(channels.keys())
    intensity_data = {ch: df_filtered[f'Mean_Intensity_{ch}'].tolist() for ch in channel_order} if not df_filtered.empty else {ch: [] for ch in channel_order}
    
    reconstructed_image_bytes = reconstruct_image(channels, seg_channel_name, df_filtered, masks_dict, tile_metadata)

    plots_data = {
        'reconstruction': reconstructed_image_bytes,
        'cell_count': create_cell_count_plot(initial_cell_count, final_cell_count).getvalue(),
        'area_histogram': create_area_histogram(df_filtered).getvalue(),
        'histogram': create_histogram(intensity_data, channel_order).getvalue(),
        'boxplot': create_boxplot(intensity_data, channel_order).getvalue()
    }
    
    tasks[task_id]['data'] = df_filtered
    tasks[task_id]['plots_data'] = plots_data
    tasks[task_id]['result'] = {'plot_names': list(plots_data.keys()), 'barChartData': {}}
    tasks[task_id]['status'] = 'Complete'; tasks[task_id]['progress'] = 100

def run_comparison_mode(task_id, channels, tile_metadata, thresholds):
    channel_results, model = {}, Cellpose(model_type='cyto', gpu=True)
    for i, (ch_name, ch_tiles) in enumerate(channels.items()):
        tasks[task_id]['status'] = f'Analyzing channel {i+1}/{len(channels)}: {ch_name}...'; tasks[task_id]['progress'] = 10 + int(80 * (i / len(channels)))
        
        all_cells_data = []
        for tile in ch_tiles:
            img_raw = cv2.imdecode(np.frombuffer(tile['data'], np.uint8), cv2.IMREAD_GRAYSCALE)
            masks, _, _, _ = model.eval(img_raw, diameter=None, channels=[0,0])
            for cell_id_on_tile in np.unique(masks)[1:]:
                cell_mask = (masks == cell_id_on_tile)
                M = cv2.moments(cell_mask.astype(np.uint8))
                if M["m00"] > 0:
                    local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    o = tile_metadata.get('overlap', 0)
                    if not (tile['x'] > 0 and local_cx < o) and not (tile['y'] > 0 and local_cy < o):
                        all_cells_data.append({'Area_pixels': int(np.sum(cell_mask)), f'Mean_Intensity_{ch_name}': float(np.mean(img_raw[cell_mask]))})

        df = pd.DataFrame(all_cells_data) if all_cells_data else pd.DataFrame()
        df_filtered = df.copy()
        if not df.empty:
            df_filtered = df[(df['Area_pixels'] >= thresholds['min_area']) & (df['Area_pixels'] <= thresholds['max_area'])]
            intensity_col = f'Mean_Intensity_{ch_name}'
            if intensity_col in df_filtered:
                df_filtered = df_filtered[(df_filtered[intensity_col] >= thresholds['min_intensity']) & (df_filtered[intensity_col] <= thresholds['max_intensity'])]
        
        channel_results[ch_name] = {'count': len(df_filtered), 'df': df_filtered}

    tasks[task_id]['status'] = 'Creating final plots...'; tasks[task_id]['progress'] = 95
    final_df_list = []
    for ch_name, res in channel_results.items():
        res['df']['Channel'] = ch_name
        final_df_list.append(res['df'])
    final_df = pd.concat(final_df_list, ignore_index=True) if final_df_list else pd.DataFrame()

    plots_data = {'channel_comparison': create_channel_comparison_plot(channel_results).getvalue()}
    
    tasks[task_id]['status'] = 'Complete'; tasks[task_id]['progress'] = 100
    tasks[task_id]['result'] = {'plot_names': list(plots_data.keys()), 'barChartData': {}}
    tasks[task_id]['data'] = final_df
    tasks[task_id]['plots_data'] = plots_data

def process_analysis_request(task_id, files_with_paths, thresholds, bg_coords, comp_matrix):
    try:
        tasks[task_id]['status'] = 'Grouping tiles...'; tasks[task_id]['progress'] = 5
        channels, tile_metadata = {}, {}
        coord_pattern = re.compile(r'tile_y(\d+)_x(\d+)\.png')
        for path, file_data in files_with_paths.items():
            path_parts = path.split('/');
            if len(path_parts) < 2: continue
            channel_name, filename = path_parts[-2], path_parts[-1]
            if channel_name not in channels: channels[channel_name] = []
            match = coord_pattern.search(filename)
            if not match: continue
            tile_info = {'data': file_data, 'y': int(match.group(1)), 'x': int(match.group(2))}
            channels[channel_name].append(tile_info)
            if 'tile_size' not in tile_metadata:
                temp_img = cv2.imdecode(np.frombuffer(tile_info['data'], np.uint8), cv2.IMREAD_GRAYSCALE)
                if temp_img is not None:
                    tile_metadata['tile_size'] = temp_img.shape[0]; tile_metadata['overlap'] = temp_img.shape[0] // 5
        
        if thresholds['analysis_mode'] == 'quantify':
            run_quantification_mode(task_id, channels, tile_metadata, thresholds, bg_coords, comp_matrix)
        elif thresholds['analysis_mode'] == 'compare':
            run_comparison_mode(task_id, channels, tile_metadata, thresholds)
        else:
            raise ValueError("Invalid analysis mode selected.")
    except Exception as e:
        tasks[task_id]['status'] = 'Error'; tasks[task_id]['error'] = str(e); traceback.print_exc()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'Pending...', 'progress': 0}
    files_with_paths = {file.filename: file.read() for file in request.files.getlist('images')}
    thresholds = {
        'analysis_mode': request.form['analysisMode'], 'seg_channel': request.form.get('segChannel'),
        'min_area': float(request.form.get('minArea', 0)), 'max_area': float(request.form.get('maxArea', 1e9)),
        'min_intensity': float(request.form.get('minIntensity', 0)), 'max_intensity': float(request.form.get('maxIntensity', 1e9))
    }
    bg_coords = {'x1': int(float(request.form.get('bgX1',0))), 'y1': int(float(request.form.get('bgY1',0))),
                 'x2': int(float(request.form.get('bgX2',0))), 'y2': int(float(request.form.get('bgY2',0)))}
    comp_matrix = np.identity(len(files_with_paths))
    
    thread = Thread(target=process_analysis_request, args=(task_id, files_with_paths, thresholds, bg_coords, comp_matrix))
    thread.daemon = True
    thread.start()
    return jsonify({'success': True, 'task_id': task_id})

@app.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if task is None: return jsonify({'status': 'Error', 'error': 'Task ID not found.'})
    safe_task_info = {'status': task.get('status'), 'progress': task.get('progress'), 'result': task.get('result')}
    if 'error' in task: safe_task_info['error'] = task['error']
    return jsonify(safe_task_info)

@app.route('/get_plot/<task_id>/<plot_name>')
def get_plot(task_id, plot_name):
    task = tasks.get(task_id, {})
    if 'plots_data' in task and plot_name in task['plots_data']:
        return send_file(io.BytesIO(task['plots_data'][plot_name]), mimetype='image/png')
    return "Plot not found", 404

@app.route('/download_csv/<task_id>')
def download_csv(task_id):
    task = tasks.get(task_id, {})
    if 'data' in task and task['data'] is not None:
        df = task['data']; csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mem_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8')); mem_buffer.seek(0)
        return send_file(mem_buffer, as_attachment=True, attachment_filename='analysis_results.csv', mimetype='text/csv')
    return "No data to download", 404

@app.route('/download_plots_pdf/<task_id>')
def download_plots_pdf(task_id):
    task = tasks.get(task_id, {})
    if 'plots_data' in task:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zf:
            for name, plot_data_bytes in task['plots_data'].items():
                zf.writestr(f'{name}.png', plot_data_bytes)
        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, attachment_filename='analysis_results.zip', mimetype='application/zip')
    return "No plots to download", 404

if __name__ == '__main__':
    app.run(debug=True)
