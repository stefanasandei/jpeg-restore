from models.adm_unet import ADMRestoration
from models.dimsum import DiMSUMRestoration
from models.fbcnn import FBCNN
from models.linear_dit import LinearDiTRestoration
from models.restormer import RestormerRestoration
from models.wavelet import HaarWaveletRestoration
from models.wrappers import DiMSUMRectifiedFlow

MODELS = {
    "fbcnn": FBCNN,
    "restormer": RestormerRestoration,
    "adm_unet": ADMRestoration,
    "dimsum": DiMSUMRestoration,
    "linear_dit": LinearDiTRestoration,
}
