
from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus

if __name__ == "__main__":
    arch = ResUnetPlusPlus(input_size=256)
    model = arch.build_model()
    model.summary()
