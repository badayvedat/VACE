# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class VaceImageProcessor(object):
    def __init__(self, downsample=None, seq_len=None):
        self.downsample = downsample
        self.seq_len = seq_len

    def _pillow_convert(self, image, cvt_type='RGB'):
        if image.mode != cvt_type:
            if image.mode == 'P':
                image = image.convert(f'{cvt_type}A')
            if image.mode == f'{cvt_type}A':
                bg = Image.new(cvt_type,
                               size=(image.width, image.height),
                               color=(255, 255, 255))
                bg.paste(image, (0, 0), mask=image)
                image = bg
            else:
                image = image.convert(cvt_type)
        return image

    def _load_image(self, img_path):
        if img_path is None or img_path == '':
            return None
        img = Image.open(img_path)
        img = self._pillow_convert(img)
        return img

    def _resize_crop(self, img, oh, ow, normalize=True):
        """
        Resize, center crop, convert to tensor, and normalize.
        """
        # resize and crop
        iw, ih = img.size
        if iw != ow or ih != oh:
            # resize
            scale = max(ow / iw, oh / ih)
            img = img.resize(
                (round(scale * iw), round(scale * ih)),
                resample=Image.Resampling.LANCZOS
            )
            assert img.width >= ow and img.height >= oh

            # center crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))

        # normalize
        if normalize:
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(1)
        return img
    
    def _image_preprocess(self, img, oh, ow, normalize=True, **kwargs):
        return self._resize_crop(img, oh, ow, normalize)

    def load_image(self, data_key, **kwargs):
        return self.load_image_batch(data_key, **kwargs)

    def load_image_pair(self, data_key, data_key2, **kwargs):
        return self.load_image_batch(data_key, data_key2, **kwargs)

    def load_image_batch(self, *data_key_batch, normalize=True, seq_len=None, **kwargs):
        seq_len = self.seq_len if seq_len is None else seq_len
        imgs = []
        for data_key in data_key_batch:
            img = self._load_image(data_key)
            imgs.append(img)
        w, h = imgs[0].size
        dh, dw = self.downsample[1:]

        # compute output size
        scale = min(1., np.sqrt(seq_len / ((h / dh) * (w / dw))))
        oh = int(h * scale) // dh * dh
        ow = int(w * scale) // dw * dw
        assert (oh // dh) * (ow // dw) <= seq_len
        imgs = [self._image_preprocess(img, oh, ow, normalize) for img in imgs]
        return *imgs, (oh, ow)


class VaceVideoProcessor(object):
    def __init__(self, downsample, min_area, max_area, min_fps, max_fps, zero_start, seq_len, keep_last, **kwargs):
        self.downsample = downsample
        self.min_area = min_area
        self.max_area = max_area
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.zero_start = zero_start
        self.keep_last = keep_last
        self.seq_len = seq_len
        assert seq_len >= min_area / (self.downsample[1] * self.downsample[2])

    def set_area(self, area):
        self.min_area = area
        self.max_area = area

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len

    @staticmethod
    def resize_crop(video: torch.Tensor, oh: int, ow: int):
        """
        Resize, center crop and normalize for decord loaded video (torch.Tensor type)

        Parameters:
          video - video to process (torch.Tensor): Tensor from `reader.get_batch(frame_ids)`, in shape of (T, H, W, C)
          oh - target height (int)
          ow - target width (int)

        Returns:
            The processed video (torch.Tensor): Normalized tensor range [-1, 1], in shape of (C, T, H, W)

        Raises:
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video

    def _video_preprocess(self, video, oh, ow):
        return self.resize_crop(video, oh, ow)

    def _get_frameid_bbox_default(self, fps, frame_timestamps, h, w, crop_box, rng):
        target_fps = min(fps, self.max_fps)
        duration = frame_timestamps[-1, 1]
        
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        df, dh, dw = self.downsample
        
        # min/max area of the [latent video]
        min_area_z = self.min_area / (dh * dw)
        max_area_z = min(self.seq_len, self.max_area / (dh * dw), (h // dh) * (w // dw))
        if min_area_z > max_area_z:
            min_area_z, max_area_z = max_area_z, min_area_z
        
        # Sample a frame number of the [latent video]
        rand_area_z = np.square(np.power(2, rng.uniform(
            np.log2(np.sqrt(min_area_z)),
            np.log2(np.sqrt(max_area_z))
        )))
        
        # Calculate frame count, ensuring it's at least 1
        frames_at_target_fps = max(1, int(duration * target_fps))
        of = min(
            (frames_at_target_fps - 1) // df + 1,
            int(self.seq_len / rand_area_z)
        )
        of = max(1, of)  # Ensure at least 1 frame
        
        # Calculate target shape
        target_area_z = min(max_area_z, int(self.seq_len / of))
        oh = round(np.sqrt(target_area_z * ratio))
        ow = int(target_area_z / oh)
        oh = max(1, oh)
        ow = max(1, ow)
        
        of = (of - 1) * df + 1
        oh *= dh
        ow *= dw
        
        # Handle special case for very short videos
        if len(frame_timestamps) <= 1:
            return [0], (x1, x2, y1, y2), (oh, ow), target_fps
        
        # Sample frame IDs
        target_duration = of / target_fps
        target_duration = min(target_duration, duration)
        
        if self.zero_start:
            begin = 0.
        else:
            max_begin = max(0., duration - target_duration)
            begin = rng.uniform(0, max_begin) if max_begin > 0 else 0.
        
        if of == 1:
            # Use middle frame for single-frame output
            frame_ids = [len(frame_timestamps) // 2]
        else:
            # Calculate timestamps and find nearest frames
            timestamps = np.linspace(begin, begin + target_duration, of)
            frame_ids = []
            
            for ts in timestamps:
                # Find the closest frame based on timestamp
                frame_idx = int(np.clip(ts * fps, 0, len(frame_timestamps) - 1))
                frame_ids.append(frame_idx)
        
        return frame_ids, (x1, x2, y1, y2), (oh, ow), target_fps
    
    def _get_frameid_bbox_adjust_last(self, fps, frame_timestamps, h, w, crop_box, rng):
        duration = frame_timestamps[-1, 1]
        
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        df, dh, dw = self.downsample
        
        # min/max area of the [latent video]
        min_area_z = self.min_area / (dh * dw)
        max_area_z = min(self.seq_len, self.max_area / (dh * dw), (h // dh) * (w // dw))
        if min_area_z > max_area_z:
            min_area_z, max_area_z = max_area_z, min_area_z
        
        # Sample a frame number
        rand_area_z = np.square(np.power(2, rng.uniform(
            np.log2(np.sqrt(min_area_z)),
            np.log2(np.sqrt(max_area_z))
        )))
        
        # Calculate frame count, ensuring it's at least 1
        num_frames = len(frame_timestamps)
        max_frames = (num_frames - 1) // df + 1
        desired_frames = int(self.seq_len / rand_area_z)
        of = min(max_frames, desired_frames)
        of = max(1, of)  # Ensure at least 1 frame
        
        # Calculate target shape
        target_area_z = min(max_area_z, int(self.seq_len / of))
        oh = round(np.sqrt(target_area_z * ratio))
        ow = int(target_area_z / oh)
        oh = max(1, oh)
        ow = max(1, ow)
        
        of = (of - 1) * df + 1
        oh *= dh
        ow *= dw
        
        # Handle special case for very short videos
        if num_frames <= 1:
            return [0], (x1, x2, y1, y2), (oh, ow), fps
        
        # Calculate target FPS and sample frames
        target_duration = duration
        target_fps = of / target_duration
        
        if of == 1:
            # Use middle frame for single-frame output
            frame_ids = [num_frames // 2]
        else:
            # Calculate evenly spaced frames
            frame_ids = np.linspace(0, num_frames - 1, of, dtype=int).tolist()
        
        return frame_ids, (x1, x2, y1, y2), (oh, ow), target_fps


    def _get_frameid_bbox(self, fps, frame_timestamps, h, w, crop_box, rng):
        if self.keep_last:
            return self._get_frameid_bbox_adjust_last(fps, frame_timestamps, h, w, crop_box, rng)
        else:
            return self._get_frameid_bbox_default(fps, frame_timestamps, h, w, crop_box, rng)

    def load_video(self, data_key, crop_box=None, seed=2024, **kwargs):
        return self.load_video_batch(data_key, crop_box=crop_box, seed=seed, **kwargs)

    def load_video_pair(self, data_key, data_key2, crop_box=None, seed=2024, **kwargs):
        return self.load_video_batch(data_key, data_key2, crop_box=crop_box, seed=seed, **kwargs)
    def load_video_batch(self, *data_key_batch, crop_box=None, seed=2024, **kwargs):
        rng = np.random.default_rng(seed)
        
        # Read videos using OpenCV
        import cv2
        import torch
        videos_data = []
        video_lengths = []
        
        for data_k in data_key_batch:
            cap = cv2.VideoCapture(data_k)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {data_k}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # Handle invalid FPS
                fps = 30.0  # Assume a reasonable default
            
            total_frames = 0
            frames = []
            success = True
            
            # Read all frames to ensure accurate count
            while success:
                success, frame = cap.read()
                if success:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    total_frames += 1
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames could be read from: {data_k}")
            
            video_lengths.append(total_frames)
            videos_data.append(frames)
        
        # Get the minimum number of frames across all videos
        length = min(video_lengths)
        
        # Get dimensions from the first frame
        h, w = videos_data[0][0].shape[:2]
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        
        # Calculate some basic dimensions
        df, dh, dw = self.downsample
        min_area_z = self.min_area / (dh * dw)
        max_area_z = min(self.seq_len, self.max_area / (dh * dw), ((y2-y1) // dh) * ((x2-x1) // dw))
        if min_area_z > max_area_z:
            min_area_z, max_area_z = max_area_z, min_area_z
        ratio = (y2 - y1) / (x2 - x1)
        
        # For very short videos, use all frames without skipping
        if length <= self.seq_len // min_area_z:  # If video is short enough to use all frames
            # Calculate the maximum possible frames after downsampling
            max_possible_frames = (length - 1) // df + 1
            
            # Calculate target shape for this number of frames
            target_area_z = min(max_area_z, int(self.seq_len / max_possible_frames))
            oh = round(np.sqrt(target_area_z * ratio)) * dh
            ow = round(target_area_z / (oh / dh)) * dw
            oh = max(dh, oh)  # Ensure at least one latent pixel
            ow = max(dw, ow)  # Ensure at least one latent pixel
            
            # Use all frames (or downsample if df > 1)
            frame_ids = []
            for i in range(0, length, df):
                frame_ids.append(i)
            
            # If we ended up with no frames, take at least one
            if not frame_ids:
                frame_ids = [0]
            
            # Set target_fps based on original fps and df
            target_fps = fps / df
        else:
            # For longer videos, use the normal frame selection logic
            # Calculate frame timestamps
            duration = length / fps
            frame_timestamps = np.zeros((length, 2), dtype=np.float32)
            for i in range(length):
                start_time = i / fps
                end_time = (i + 1) / fps
                frame_timestamps[i, 0] = start_time
                frame_timestamps[i, 1] = end_time
            
            # Use the existing frame selection logic
            frame_ids, (x1, x2, y1, y2), (oh, ow), target_fps = self._get_frameid_bbox(fps, frame_timestamps, h, w, crop_box, rng)
        
        # Ensure all frame IDs are within bounds
        frame_ids = [min(max(0, fid), length-1) for fid in frame_ids]
        
        # Extract selected frames and preprocess
        videos = []
        for video_frames in videos_data:
            selected_frames = [video_frames[fid][y1:y2, x1:x2] for fid in frame_ids]
            # Stack frames and convert to torch tensor
            frames_array = np.stack(selected_frames, axis=0)
            frames_tensor = torch.from_numpy(frames_array)
            processed_video = self._video_preprocess(frames_tensor, oh, ow)
            videos.append(processed_video)
        
        return *videos, frame_ids, (oh, ow), target_fps


def prepare_source(src_video, src_mask, src_ref_images, num_frames, image_size, device):
    for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
        if sub_src_video is None and sub_src_mask is None:
            src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
            src_mask[i] = torch.ones((1, num_frames, image_size[0], image_size[1]), device=device)
    for i, ref_images in enumerate(src_ref_images):
        if ref_images is not None:
            for j, ref_img in enumerate(ref_images):
                if ref_img is not None and ref_img.shape[-2:] != image_size:
                    canvas_height, canvas_width = image_size
                    ref_height, ref_width = ref_img.shape[-2:]
                    white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                    scale = min(canvas_height / ref_height, canvas_width / ref_width)
                    new_height = int(ref_height * scale)
                    new_width = int(ref_width * scale)
                    resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                    top = (canvas_height - new_height) // 2
                    left = (canvas_width - new_width) // 2
                    white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                    src_ref_images[i][j] = white_canvas
    return src_video, src_mask, src_ref_images
