#!/usr/bin/env python3
import os
import cv2
import xml.etree.ElementTree as ET
import glob
import multiprocessing as mp
import getpass
import argparse

# Define colors for different nerve labels
LABEL_COLORS = {
    'C5': (255, 0, 0),      # Red
    'C6': (0, 255, 0),      # Green
    'C7': (0, 0, 255),      # Blue
    'C8': (255, 255, 0),    # Cyan
    'UT': (255, 0, 255),    # Magenta
    'MT': (0, 255, 255),    # Yellow
    'LT': (128, 0, 128),    # Purple
    'SSN': (255, 165, 0),   # Orange
    'AD': (0, 128, 0),      # Dark Green
    'PD': (0, 0, 128),      # Dark Blue
}

def find_video_file(annotation_path, video_dir="/home/nistring/object-detection-project/data/raw/videos"):
    """Find the corresponding video file for an annotation"""
    # Get the directory name (which should match video name)
    dir_name = os.path.basename(os.path.dirname(annotation_path))
    
    # Look for video files in the videos directory
    
    # Try different extensions
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_path = os.path.join(video_dir, dir_name + ext)
        if os.path.exists(video_path):
            return video_path
    
    # If not found in main videos dir, search recursively
    for root, dirs, files in os.walk(os.path.dirname(video_dir)):
        for file in files:
            if file.startswith(dir_name) and file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return os.path.join(root, file)
    
    return None

def parse_annotation_file(annotation_path):
    """Parse annotation XML file and return bounding boxes by frame"""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get video info - handle both task and job structures
        size_element = root.find('.//task/size')
        if size_element is None:
            size_element = root.find('.//job/size')
            
        if size_element is None:
            print(f"No size element found in {annotation_path}")
            return None, None, None
        
        video_length = int(size_element.text)
        
        # Get original size
        width_elem = root.find('.//original_size/width')
        height_elem = root.find('.//original_size/height')
        if width_elem is not None and height_elem is not None:
            original_width = int(width_elem.text)
            original_height = int(height_elem.text)
        else:
            original_width, original_height = 960, 680  # Default
        
        # Parse tracks and boxes
        frames_data = {}
        tracks = root.findall('.//track')
        
        for track in tracks:
            label = track.get('label', 'Unknown')
            track_id = track.get('id', '0')
            
            boxes = track.findall('.//box')
            for box in boxes:
                frame_num = int(box.get('frame'))
                
                if frame_num not in frames_data:
                    frames_data[frame_num] = []
                
                # Get bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                frames_data[frame_num].append({
                    'label': label,
                    'track_id': track_id,
                    'bbox': (xtl, ytl, xbr, ybr)
                })
        
        return frames_data, video_length, (original_width, original_height)
    
    except Exception as e:
        print(f"Error parsing {annotation_path}: {e}")
        return None, None, None

def create_visualization_video(annotation_path, output_dir, video_dir="/home/nistring/object-detection-project/data/raw/videos"):
    """Create a video with bounding box visualizations and return statistics"""
    # Parse annotations
    frames_data, video_length, (orig_w, orig_h) = parse_annotation_file(annotation_path)
    if frames_data is None:
        return None
    
    # Find corresponding video file
    video_path = find_video_file(annotation_path, video_dir)
    if video_path is None:
        return None
    
    # Calculate coverage statistics
    frames_with_boxes = len(frames_data)
    coverage_ratio = frames_with_boxes / video_length if video_length > 0 else 0
    total_boxes = sum(len(boxes) for boxes in frames_data.values())
    
    # Count boxes by label
    label_stats = {}
    label_frame_counts = {}
    
    for frame_num, boxes in frames_data.items():
        labels_in_frame = set()
        for box in boxes:
            label = box['label']
            if label not in label_stats:
                label_stats[label] = 0
                label_frame_counts[label] = 0
            label_stats[label] += 1
            labels_in_frame.add(label)
        
        # Count frames where each label appears
        for label in labels_in_frame:
            label_frame_counts[label] += 1
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    file_name = os.path.basename(os.path.dirname(annotation_path))
    output_path = os.path.join(output_dir, f"{file_name}_annotated.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes if they exist for this frame
        if frame_count in frames_data:
            for box_data in frames_data[frame_count]:
                label = box_data['label']
                bbox = box_data['bbox']
                
                # Scale coordinates if needed
                scale_x = frame_width / orig_w
                scale_y = frame_height / orig_h
                
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                # Get color for this label
                color = LABEL_COLORS.get(label, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{label}_{box_data['track_id']}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), 
                            (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add density info
        boxes_in_frame = len(frames_data.get(frame_count, []))
        cv2.putText(frame, f"Boxes: {boxes_in_frame}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add coverage info
        cv2.putText(frame, f"Coverage: {coverage_ratio:.1%}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Return statistics
    stats = {
        'file_name': file_name,
        'video_length': video_length,
        'frames_with_boxes': frames_with_boxes,
        'coverage_ratio': coverage_ratio,
        'total_boxes': total_boxes,
        'density': total_boxes / video_length if video_length > 0 else 0,
        'label_stats': label_stats,
        'label_frame_counts': label_frame_counts,
        'label_coverage_ratios': {label: count / video_length for label, count in label_frame_counts.items()}
    }
    
    return stats

def analyze_annotation_file(file_path):
    """Analyze a single annotation file and return basic statistics"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get video size (length) - handle both task and job structures
        size_element = root.find('.//task/size')
        if size_element is None:
            size_element = root.find('.//job/size')
        
        if size_element is None:
            print(f"No size element found in {file_path}")
            return None
        
        video_length = int(size_element.text)
        
        # Get username from annotation file
        username_element = root.find('.//owner/username')
        username = username_element.text if username_element is not None else 'Unknown'
        
        # Count total number of bounding boxes
        boxes = root.findall('.//box')
        num_boxes = len(boxes)
        
        # Calculate density (boxes per frame)
        density = num_boxes / video_length if video_length > 0 else 0
        
        return {
            'file_path': file_path,
            'video_length': video_length,
            'num_boxes': num_boxes,
            'density': density,
            'file_name': os.path.basename(os.path.dirname(file_path)),
            'username': username
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def analyze_annotation_file_detailed(file_path):
    """Analyze a single annotation file and return detailed statistics without video processing"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get video size (length) - handle both task and job structures
        size_element = root.find('.//task/size')
        if size_element is None:
            size_element = root.find('.//job/size')
        
        if size_element is None:
            print(f"No size element found in {file_path}")
            return None
        
        video_length = int(size_element.text)
        
        # Get username from annotation file
        username_element = root.find('.//owner/username')
        username = username_element.text if username_element is not None else 'Unknown'
        
        # Parse tracks and boxes for detailed analysis
        frames_data = {}
        tracks = root.findall('.//track')
        
        for track in tracks:
            label = track.get('label', 'Unknown')
            track_id = track.get('id', '0')
            
            boxes = track.findall('.//box')
            for box in boxes:
                frame_num = int(box.get('frame'))
                
                if frame_num not in frames_data:
                    frames_data[frame_num] = []
                
                # Get bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                frames_data[frame_num].append({
                    'label': label,
                    'track_id': track_id,
                    'bbox': (xtl, ytl, xbr, ybr)
                })
        
        # Calculate detailed statistics
        frames_with_boxes = len(frames_data)
        coverage_ratio = frames_with_boxes / video_length if video_length > 0 else 0
        total_boxes = sum(len(boxes) for boxes in frames_data.values())
        density = total_boxes / video_length if video_length > 0 else 0
        
        # Count boxes by label
        label_stats = {}
        label_frame_counts = {}
        
        for frame_num, boxes in frames_data.items():
            labels_in_frame = set()
            for box in boxes:
                label = box['label']
                if label not in label_stats:
                    label_stats[label] = 0
                    label_frame_counts[label] = 0
                label_stats[label] += 1
                labels_in_frame.add(label)
            
            # Count frames where each label appears
            for label in labels_in_frame:
                label_frame_counts[label] += 1
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(os.path.dirname(file_path)),
            'video_length': video_length,
            'frames_with_boxes': frames_with_boxes,
            'coverage_ratio': coverage_ratio,
            'total_boxes': total_boxes,
            'density': density,
            'username': username,
            'label_stats': label_stats,
            'label_frame_counts': label_frame_counts,
            'label_coverage_ratios': {label: count / video_length for label, count in label_frame_counts.items()}
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_single_video(args):
    """Process a single video with annotations - designed for multiprocessing"""
    result, output_dir, video_dir, total_count, current_index = args
    
    if current_index % 10 == 0:  # Only print every 10th file
        print(f"[{current_index}/{total_count}] Processing: {result['file_name']}")
    
    return create_visualization_video(result['file_path'], output_dir, video_dir)
def create_summary_and_csv(all_stats, output_dir, username, skip_video_processing=False):
    """Create simple summary and CSV files"""
    import datetime
    
    # Basic stats
    total_videos = len(all_stats)
    total_boxes = sum(s['total_boxes'] for s in all_stats)
    avg_density = sum(s['density'] for s in all_stats) / total_videos if total_videos > 0 else 0
    
    # Simple summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Annotation Analysis Summary\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"By: {username}\n\n")
        f.write(f"Videos: {total_videos}\n")
        f.write(f"Total boxes: {total_boxes:,}\n")
        f.write(f"Avg density: {avg_density:.3f} boxes/frame\n\n")
        
        # Top files by density
        sorted_stats = sorted(all_stats, key=lambda x: x['density'], reverse=True)
        f.write("Top files by density:\n")
        for i, stats in enumerate(sorted_stats[:10]):
            f.write(f"{i+1:2d}. {stats['file_name']:<25} {stats['density']:.3f}\n")
    
    # CSV
    csv_path = os.path.join(output_dir, "annotation_statistics.csv")
    with open(csv_path, 'w') as f:
        f.write("file_name,video_length,total_boxes,density,coverage_ratio,username\n")
        for stats in all_stats:
            f.write(f"{stats['file_name']},{stats['video_length']},{stats['total_boxes']},")
            f.write(f"{stats['density']:.4f},{stats['coverage_ratio']:.4f},{stats.get('username', 'Unknown')}\n")
    
    print(f"Summary saved: {summary_path}")
    print(f"CSV saved: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize annotations and generate statistics')
    parser.add_argument('--stats-only', action='store_true', help='Skip video processing')
    parser.add_argument('--username', type=str, default=getpass.getuser(), help='Username for reports')
    parser.add_argument('--input-dir', type=str, help='Input directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--video-dir', type=str, help='Video directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    annotation_files = glob.glob(f"{args.input_dir}/**/annotations.xml", recursive=True)
    
    print(f"Processing {len(annotation_files)} annotation files...")
    
    if args.stats_only:
        all_stats = [analyze_annotation_file_detailed(f) for f in annotation_files]
        all_stats = [s for s in all_stats if s]
    else:
        results = [analyze_annotation_file(f) for f in annotation_files]
        results = [r for r in results if r]
        results.sort(key=lambda x: x['density'], reverse=True)
        
        process_args = [(r, args.output_dir, args.video_dir, len(results), i+1) 
                       for i, r in enumerate(results)]
        
        with mp.Pool(max(1, int(mp.cpu_count() * 0.9))) as pool:
            all_stats = pool.map(process_single_video, process_args)
        all_stats = [s for s in all_stats if s]
    
    create_summary_and_csv(all_stats, args.output_dir, args.username, args.stats_only)
    print(f"Done! Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
