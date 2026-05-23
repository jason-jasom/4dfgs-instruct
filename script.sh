echo "Editing with prompt: Make it look like a Fauvism painting"
python edit_3d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud/iteration_120000 \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/painting \
  --edited_pattern "edited_painting_original_time{frame_id}_{camera_id}.png" \
  --prompt "Make it look like a fauvism painting" \
  --iterations 1000 \
  --save_iterations 500 1000 \
  --anchor_update \
  --start_stat 100 \
  --update_from 200 \
  --update_interval 50 \
  --update_until 801 \
  --freeze_mlp

python refine_sds.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000" \
  --prompt "Make it look like a fauvism painting" \
  --iterations 800 \
  --disable_anchor_update \
  --freeze_mlp

python render_edited4d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_refine/Make it look like a fauvism painting/iteration_800" \
  --output_name painting_refine_800 \
  --video_name painting.mp4 \
  --skip_train

echo "Editing with prompt: Make it look like a sculpture"

python edit_3d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud/iteration_120000 \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/sculpture \
  --edited_pattern "edited_sculpture_original_time{frame_id}_{camera_id}.png" \
  --prompt "Make it look like a sculpture" \
  --iterations 1000 \
  --save_iterations 500 1000 \
  --anchor_update \
  --start_stat 100 \
  --update_from 200 \
  --update_interval 50 \
  --update_until 801 \
  --freeze_mlp

python refine_sds.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_3dedit/Make it look like a sculpture/iteration_1000" \
  --prompt "Make it look like a sculpture" \
  --iterations 800 \
  --disable_anchor_update \
  --freeze_mlp

python render_edited4d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_refine/Make it look like a sculpture/iteration_800" \
  --output_name sculpture_refine_800 \
  --video_name sculpture.mp4 \
  --skip_train

echo "Editing with prompt: Turn the man into a woman"

python edit_3d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud/iteration_120000 \
  --edited_images_path /media/ai2lab/SSD4TB/EV_final/data/N3DV/cook_spinach/woman \
  --edited_pattern "edited_woman_original_time{frame_id}_{camera_id}.png" \
  --prompt "Turn the man into a woman" \
  --iterations 1000 \
  --save_iterations 500 1000 \
  --anchor_update \
  --start_stat 100 \
  --update_from 200 \
  --update_interval 50 \
  --update_until 801 \
  --freeze_mlp

python refine_sds.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_3dedit/Turn the man into a woman/iteration_1000" \
  --prompt "Turn the man into a woman" \
  --iterations 800 \
  --disable_anchor_update \
  --freeze_mlp

python render_edited4d.py \
  -m outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11 \
  --checkpoint_dir "outputs/N3DV/cook_spinach/baseline/2026-05-12_15-56-11/point_cloud_refine/Turn the man into a woman/iteration_800" \
  --output_name woman_refine_800 \
  --video_name woman.mp4 \
  --skip_train
