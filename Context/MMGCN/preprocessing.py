

import freerec


freerec.declare(version="0.7.3")


cfg = freerec.parser.Parser()

cfg.add_argument("-di", "--download-images", action="store_true", default=False)
cfg.add_argument("--image-folder", type=str, default="item_images")
cfg.add_argument("--image-size", type=str, choices=('thumb', 'large', 'hi_res'), default="thumb")

cfg.add_argument("--model_dir", type=str, default="../../models")

cfg.add_argument("-rvm", "--require-visual-modality", action="store_true", default=False)
cfg.add_argument("--visual-model", type=str, default="Ramos-Ramos/dino-resnet-50")
cfg.add_argument("--visual-saved_file", type=str, default="visual_modality.pkl")

cfg.add_argument("-rtm", "--require-textual-modality", action="store_true", default=False)
cfg.add_argument("--textual-model", type=str, default="all-MiniLM-L6-v2")
cfg.add_argument("--textual-saved_file", type=str, default="textual_modality.pkl")
cfg.add_argument("--field-names", type=str, default="Title,Features,Description,Details")

cfg.set_defaults(
    description="MMGCN",
    root="../../data",
    dataset='Amazon2023_Toys_10100_Chron',
    batch_size=256,
)
cfg.compile()


def main():

    dataset: freerec.data.datasets.context.Amazon2023 = getattr(freerec.data.datasets.context, cfg.dataset)(cfg.root)

    if cfg.download_images:
        dataset.download_images(
            image_folder=cfg.image_folder,
            image_size=cfg.image_size
        )
    if cfg.require_visual_modality:
        dataset.encode_visual_modality(
            cfg.visual_model, model_dir=cfg.model_dir,
            image_folder=cfg.image_folder, image_size=cfg.image_size,
            saved_file=cfg.visual_saved_file, num_workers=cfg.num_workers, batch_size=cfg.batch_size
        )
    if cfg.require_textual_modality:
        dataset.encode_textual_modality(
            cfg.textual_model, model_dir=cfg.model_dir,
            field_names=cfg.field_names.split(','),
            saved_file=cfg.textual_saved_file, batch_size=cfg.batch_size
        )


if __name__ == "__main__":
    main()