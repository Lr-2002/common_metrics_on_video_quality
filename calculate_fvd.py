import numpy as np
import torch
from tqdm import tqdm

class FVD:
    def __init__(self, method='styleganv', device='cuda'):
        if method == 'styleganv':
            from .fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
        elif method == 'videogpt':
            from .fvd.videogpt.fvd import load_i3d_pretrained
            from .fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
            from .fvd.videogpt.fvd import frechet_distance
        self.get_fvd_feats = get_fvd_feats
        self.frechet_distance = frechet_distance

        self.i3d = load_i3d_pretrained(device=device)
        self.device = device

    def trans(self, x):
        # if greyscale images add channel
        if x.shape[-3] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        # permute BTCHW -> BCTHW
        x = x.permute(0, 2, 1, 3, 4)

        return x
    def __call__(self, input1, input2):
        return self.calculate_fvd(input1, input2)

    def calculate_fvd(self, videos1, videos2):
        # videos [batch_size, timestamps, channel, h, w]

        assert videos1.shape == videos2.shape

        fvd_results = []

        # support grayscale input, if grayscale -> channel*3
        # BTCHW -> BCTHW
        # videos -> [batch_size, channel, timestamps, h, w]

        videos1 = self.trans(videos1)
        videos2 = self.trans(videos2)

        fvd_results = {}

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):

            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = self.get_fvd_feats(videos_clip1, i3d=self.i3d, device=self.device)
            feats2 = self.get_fvd_feats(videos_clip2, i3d=self.i3d, device=self.device)

            # calculate FVD when timestamps[:clip]
            fvd_results[clip_timestamp] = self.frechet_distance(feats1, feats2)
        result = fvd_results

        return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_fvd(videos1, videos2, device, method='videogpt')
    print(json.dumps(result, indent=4))

    result = calculate_fvd(videos1, videos2, device, method='styleganv')
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
