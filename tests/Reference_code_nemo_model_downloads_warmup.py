# diag_list_models.py
import nemo
from nemo.collections.asr.models import ClusteringDiarizer, EncDecSpeakerLabelModel, EncDecClassificationModel

print("nemo.__version__ =", nemo.__version__)

def safe_list(cls):
    print(f"\nClass: {cls.__name__}")
    # show if the method exists
    print("Has list_available_models:", hasattr(cls, "list_available_models"))
    try:
        available = cls.list_available_models()
        print("Raw returned value:", repr(available))
        if not available:
            print("=> list_available_models returned None or empty. Possible network/index/API issue.")
            return []
        # some versions return objects; try to be robust
        names = []
        for m in available:
            # try different attributes safely
            name = getattr(m, "pretrained_model_name", None) or getattr(m, "name", None) or str(m)
            names.append(name)
        return names
    except Exception as e:
        print("list_available_models() raised:", type(e), e)
        return []

print("ClusteringDiarizer:", safe_list(ClusteringDiarizer))
print("SpeakerEmbeddingModel:", safe_list(EncDecSpeakerLabelModel))
print("VAD Model Class:", safe_list(EncDecClassificationModel))
