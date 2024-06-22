import os
from typing import Generator, List

from PIL import Image
import chromadb
from chromadb.config import Settings
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from flask import (
    Flask,
    jsonify,
    render_template,
    render_template_string,
    request,
    send_file,
)
import numpy as np
import osxphotos


IMAGE_BASE_DIR = os.path.join(
    os.environ["HOME"], "Pictures/Photos Library.photoslibrary"
)

COLLECTION_NAME = "multimodal_collection"


class Indexer:
    def __init__(self, collection) -> None:
        self.collection = collection

    def index(self):
        files = Indexer._find_photo_files(pattern=".jpeg")
        filename_array_tuple = Indexer._process_images(list(files))
        ids, arrays = zip(*filename_array_tuple)
        self.collection.add(ids=list(ids), images=list(arrays))

    @staticmethod
    def _find_photo_files(pattern: str) -> Generator[str, str, None]:
        photos = osxphotos.PhotosDB()
        for photo in photos.photos():
            if isinstance(photo.path, str) and pattern in photo.path:
                yield photo.path

    @staticmethod
    def _numpy_array_for_images(filenames: List[str]):
        for filename in filenames:
            image = Image.open(filename)
            yield (filename, np.array(image))

    @staticmethod
    def _process_images(filenames):
        for filename in filenames:
            try:
                with Image.open(filename) as image:
                    yield (filename, np.array(image))
            except OSError as e:
                print(f"Error processing {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {filename}: {e}")


class ImageSearch(Flask):
    def __init__(self, import_name, *args, **kwargs):
        super().__init__(import_name, *args, **kwargs)
        if not os.path.exists(IMAGE_BASE_DIR):
            raise FileNotFoundError(f"The directory {IMAGE_BASE_DIR} does not exist.")
        client = chromadb.PersistentClient(
            path="./chroma-db-instance", settings=Settings(anonymized_telemetry=False)
        )
        data_loader = ImageLoader()
        embedding_function = OpenCLIPEmbeddingFunction()
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            data_loader=data_loader,
            get_or_create=True,
        )
        self.client = client
        self.collection = collection
        self.indexer = Indexer(collection=self.collection)
        print("configured app")


app = ImageSearch(__name__)


@app.route("/index-photos", methods=["POST"])
def index_photos():
    app.indexer.index()
    return jsonify({"message": "Indexing complete"})


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    results = app.collection.query(query_texts=[query])
    ids = results.get("ids")[0]
    dl = results.get("distances")
    distances = dl[0] if dl else [0] * len(ids)
    zipped_results = zip(ids, distances)
    formatted_results = [
        {"photo_id": photo_id, "distance": distance}
        for photo_id, distance in zipped_results
    ]

    return jsonify({"results": formatted_results})


@app.route("/collection")
def collection_info():
    return jsonify({"count": app.collection.count()})


@app.route("/image")
def serve_image():
    image_path = request.args.get("path", "")
    if not image_path:
        return "No image path provided", 400
    split = image_path.split(IMAGE_BASE_DIR)
    if len(split) == 1:
        return "No image found", 404
    safe_path = os.path.normpath(os.path.join(IMAGE_BASE_DIR, split[1][1:]))

    if os.path.isfile(safe_path):
        return send_file(safe_path)
    else:
        return "Image not found", 404


@app.route("/health")
def health():
    print("here")
    return render_template_string("OK")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
