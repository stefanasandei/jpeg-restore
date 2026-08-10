from models.adm_unet import ADMRestoration
from models.fbcnn import FBCNN
from models.linear_dit import LinearDiTRestoration
from models.restormer import RestormerRestoration
from models.wavelet import HaarWaveletRestoration
from models.wrappers import LinearDiTPalette, MeanFlow, Palette, RectifiedFlow

MODELS = {
    "fbcnn": FBCNN,
    "restormer": RestormerRestoration,
    "adm_unet": ADMRestoration,
    "linear_dit": LinearDiTRestoration,
}
