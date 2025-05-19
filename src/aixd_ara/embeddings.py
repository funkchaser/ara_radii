from aixd.embedding.architecture.models.autoencoder import AutoEncoder
from aixd.embedding.architecture.models.clip import CLIPEncoder
from aixd.embedding.architecture.models.contrast_encoder import EmbeddingEncoder
from aixd.embedding.architecture.models.predictor import Predictor
from aixd.embedding.architecture.models.switchtab import SwitchTab
from aixd.embedding.architecture.models.sae import SparseAutoEncoder

from aixd.embedding.data.data_loader import EmbeddingDataModule
from aixd.embedding.data.selector import SelectorConfig


"""
step1: Data --> DataModule, depending on the embeddings model
* batch_size
* inputML, outputML - model-specific?
* use Selector or not

step2: Embeddings model setup:
* all need inputML,outputML, latent_dim, layer_widths
* some have additional params

step3: Fit the model
* max_epochs
* run model.fit

step4: Get embeddings
* model.forward() - extract data from dict, model-specific

step5: optional - projection
"""


MODEL_MAP = {
    "autoencoder": AutoEncoder,
    "predictor": Predictor,
    "contrast_encoder": EmbeddingEncoder,
    "switchtab": SwitchTab,
    "clip": CLIPEncoder,
    "sparse_autoencoder": SparseAutoEncoder,
}

PROJECTION_MAP = {"pca": None, "umap": None}

umap_params = {"n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"}


def embeddings_setup(dataset, datamodule_settings, model_type, model_settings):

    if model_type in ["switchtab", "contrast_encoder"]:
        selector_config = SelectorConfig(**datamodule_settings["selector"])
    datamodule_settings.upadte({"selector_config": selector_config})
    datamodule = EmbeddingDataModule.from_dataset(dataset=dataset, **datamodule_settings)

    model = MODEL_MAP[model_type].from_datamodule(datamodule, **model_settings)

    projection = None
    emb = Embedding(datamodule=datamodule, model=model, projection=projection)
    return emb


class Embedding:
    def __init__(self, datamodule, model, projection):
        self.datamodule = datamodule
        self.model = model
        self.projection = projection  # unused

    def train(self, training_settings):
        self.model.fit(self.datamodule, **training_settings)

    def embed(self, input=None):
        if not input:
            input = self.datamodule.train_dataloader()
        outputs = self.model.predict(input)
        embeddings = outputs["e_x"]  # TODO: check if this is the same for all model types
        return embeddings

    # def project(self, embeddings):
    #     umap_3d = EmbeddingPlotter.reduce(embeddings, dim=3, method="umap", **umap_params)
    #     return umap_3d

    def embed_and_project(self):
        emb = self.embed()
        return self.project(emb)
