import cv2
import os
import argparse
import json
from pathlib import Path
from player_tracker import PlayerTracker
from analysis import analyze_player_trajectories, analyze_performance_metrics, load_tracking_results
import time

def main():
    """
    Main function to run player tracking on input video
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Player Tracking and Re-identification System')
    parser.add_argument('--input_video', type=str, default='../input_video/15sec_input_720p.mp4',
                       help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='../object detection model/best.pt',
                       help='Path to YOLOv11 model file')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output files')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--save_video', action='store_true',
                       help='Save annotated output video')
    parser.add_argument('--display', action='store_true',
                       help='Display video during processing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if input files exist
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found at {args.input_video}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    print(f"Starting Player Tracking System...")
    print(f"Input Video: {args.input_video}")
    print(f"Model: {args.model_path}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 50)
    
    # Initialize the tracker
    try:
        tracker = PlayerTracker(args.model_path, args.confidence)
        print("✓ Tracker initialized successfully")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return
    
    # Open video
    try:
        cap = cv2.VideoCapture(args.input_video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.input_video}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Video loaded: {width}x{height} @ {fps}fps, {total_frames} frames")
        
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    # Set up video writer if saving output
    out_writer = None
    if args.save_video:
        output_video_path = output_dir / 'tracked_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"✓ Output video will be saved to: {output_video_path}")
    
    # Processing variables
    frame_count = 0
    tracking_data = []
    start_time = time.time()
    
    print("\n🎬 Starting video processing...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, tracking_info = tracker.process_frame(frame)
            
            # Store tracking information
            tracking_data.append(tracking_info)
            
            # Save frame if requested
            if out_writer is not None:
                out_writer.write(annotated_frame)
            
            # Display frame if requested
            if args.display:
                cv2.imshow('Player Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user")
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time
                print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Clean up
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\n✓ Processing completed in {total_time:.2f} seconds")
        print(f"✓ Processed {frame_count} frames")
    
    # Get performance metrics
    performance_metrics = tracker.get_performance_metrics()
    
    # Save tracking results
    results_file = output_dir / 'tracking_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'video_info': {
                'filename': os.path.basename(args.input_video),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': frame_count
            },
            'performance_metrics': performance_metrics,
            'tracking_data': tracking_data[-10:]  # Save last 10 frames of data
        }, f, indent=2)
    
    # Save detailed tracking data
    tracker.save_tracking_results(str(output_dir / 'detailed_tracking.pkl'))
    
    # Print summary
    print("\n" + "="*60)
    print("TRACKING SUMMARY")
    print("="*60)
    print(f"Total Unique Players Detected: {performance_metrics.get('total_unique_players', 0)}")
    print(f"Average Processing Time per Frame: {performance_metrics.get('avg_processing_time', 0):.4f}s")
    print(f"Average Detections per Frame: {performance_metrics.get('avg_detections_per_frame', 0):.1f}")
    print(f"Processing FPS: {performance_metrics.get('fps', 0):.1f}")
    print(f"Results saved to: {output_dir}")
    
    if args.save_video:
        print(f"Annotated video saved to: {output_dir}/tracked_output.mp4")
    
    # Generate trajectory analysis
    print("\n📊 Generating trajectory analysis...")
    try:
        results = load_tracking_results(str(output_dir))
        analyze_player_trajectories(results)
        analyze_performance_metrics(results)
        print("✅ Trajectory analysis saved to: trajectory_analysis.png")
        print("✅ Performance analysis saved to: performance_analysis.png")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate analysis plots: {e}")
    
    print("\n🎉 Player tracking completed successfully!")

if __name__ == "__main__":
    main() 