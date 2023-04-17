import cv2

from nos.test_utils import NOS_TEST_AUDIO, NOS_TEST_IMAGE, NOS_TEST_VIDEO


def test_image():
    img = cv2.imread(str(NOS_TEST_IMAGE))
    assert img is not None

    H, W, C = img.shape
    assert C >= 3
    assert H * W > 0


def test_video():
    video = cv2.VideoCapture(str(NOS_TEST_VIDEO))
    assert video.isOpened()

    for _ in range(10):
        ret, img = video.read()
        assert ret
        assert img is not None

        H, W, C = img.shape
        assert C >= 3
        assert H * W > 0


def test_audio():
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(NOS_TEST_AUDIO))
    assert waveform is not None

    # Check the waveform shape
    # waveform.shape == (C, L),
    # where C is the number of channels, and L is the number of samples
    assert len(waveform.shape) == 2
    C, L = waveform.shape
    assert C == 2
    assert L > 0
    assert sample_rate == 44100
