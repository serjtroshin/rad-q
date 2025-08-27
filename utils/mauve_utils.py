
from mauve.compute_mauve import get_features_from_input, compute_mauve

# from transformers import ElectraForPreTraining, ElectraTokenizerFast

CACHR_DIR="/.cache/"

# discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator")
FEATURE_MODEL_NAME = "google/electra-large-discriminator"
MAX_LEN = 512
BATCH_SIZE = 32

def _compute(
        predictions,
        references,
        p_features=None,
        q_features=None,
        p_tokens=None,
        q_tokens=None,
        num_buckets="auto",
        pca_max_data=-1,
        kmeans_explained_var=0.9,
        kmeans_num_redo=5,
        kmeans_max_iter=500,
        featurize_model_name="gpt2-large",
        device_id=-1,
        max_text_length=512,
        divergence_curve_discretization_size=25,
        mauve_scaling_factor=5,
        verbose=True,
        seed=25,
        batch_size=16,
    ):
        out = compute_mauve(
            p_text=predictions,
            q_text=references,
            p_features=p_features,
            q_features=q_features,
            p_tokens=p_tokens,
            q_tokens=q_tokens,
            num_buckets=num_buckets,
            pca_max_data=pca_max_data,
            kmeans_explained_var=kmeans_explained_var,
            kmeans_num_redo=kmeans_num_redo,
            kmeans_max_iter=kmeans_max_iter,
            featurize_model_name=featurize_model_name,
            device_id=device_id,
            max_text_length=max_text_length,
            divergence_curve_discretization_size=divergence_curve_discretization_size,
            mauve_scaling_factor=mauve_scaling_factor,
            verbose=verbose,
            seed=seed,
            batch_size=batch_size,
        )
        return out

def evaluate_mauve(predictions, references, device_id=None, verbose=False):
    mauve_results = _compute(
        predictions=predictions, 
        references=references,
        # q_features=reference_features, 
        featurize_model_name=FEATURE_MODEL_NAME,
        max_text_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        verbose=verbose,
        device_id=device_id,
        )
    return mauve_results.mauve, mauve_results
