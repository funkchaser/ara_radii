from aixd.embedding.architecture.models.autoencoder import AutoEncoder
from aixd.embedding.architecture.models.clip import CLIPEncoder
from aixd.embedding.architecture.models.contrast_encoder import EmbeddingEncoder
from aixd.embedding.architecture.models.predictor import Predictor
from aixd.embedding.architecture.models.switchtab import SwitchTab
from aixd.embedding.architecture.models.sae import SparseAutoEncoder

from aixd.embedding.data.data_loader import EmbeddingDataModule
from aixd.embedding.data.selector import SelectorConfig


from aixd.embedding.visualisation.plotter import EmbeddingPlotter

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


def embeddings_setup(dataset, settings):
    """
    settings: a dict with :
        * selector_settings
        * datamodule_settings
        * model_type
        * model_settings (kwargs)
        * projection_type
        * projection_settings (kwargs)

    return: some object that later consumes the data and returns embedded data
    """
    # TODO: replace with kwargs 'selector_settings'
    selector_config = SelectorConfig(
        selector_type="distance",
        params={"distance_type": "euclidean", "num_positives": 30, "num_negatives": 0, "lambda": None},
    )
    # TODO: replace with kwargs 'datamodule_settings'
    datamodule = EmbeddingDataModule.from_dataset(dataset, batch_size=1024, selector_config=selector_config)

    model = MODEL_MAP[settings["model_type"]].from_datamodule(datamodule=datamodule, **settings["model_settings"])

    projection = None  # PROJECTION_MAP[settings['projection_type']](**settings['projection_settings'])

    emb = Embedding(datamodule=datamodule, model=model, projection=projection)
    return emb


class Embedding:
    def __init__(self, datamodule, model, projection):
        self.datamodule = datamodule
        self.model = model
        self.projection = projection  # unused

    def train_model(self):
        # TODO: replace settings with kwargs
        self.model.fit(self.datamodule, max_epochs=50, flag_early_stop=True)

    def embed(self, input=None):
        if not input:
            input = self.datamodule.train_dataloader()
        outputs = self.model.predict(input)
        embeddings = outputs["e_x"]  # TODO: check if this is the same for all model types
        return embeddings

    def project(self, embeddings):
        umap_3d = EmbeddingPlotter.reduce(embeddings, dim=3, method="umap", **umap_params)
        return umap_3d

    def embed_and_project(self):
        emb = self.embed()
        return self.project(emb)
