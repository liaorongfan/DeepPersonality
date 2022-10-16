from engine.excitation_core import contrastive_excitation_backprop
import torchvision
from PIL import Image
from dpcv.tools.common import get_device
from dpcv.modeling.networks.excitation_bp_rnn import AlexNetLSTM
from dpcv.tools.draw import plot_example


def get_example_data(shape=224):
    """Get example data to demonstrate visualization techniques.

    Args:
        shape (int or tuple of int, optional): shape to resize input image to.
            Default: ``224``.

    Returns:
        (:class:`torch.nn.Module`, :class:`torch.Tensor`, int, int): a tuple
        containing

            - a convolutional neural network model in evaluation mode.
            - a sample input tensor image.
            - the ImageNet category id of an object in the image.
            - the ImageNet category id of another object in the image.

    """
    model = AlexNetLSTM()
    # Switch to eval mode to make the visualization deterministic.
    model.eval()

    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Download an example image from wikimedia.

    img = Image.open("../datasets/demo/both.png")

    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(shape),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = transform(img).unsqueeze(0)

    # bulldog category id.
    category_id_1 = 1  # 245

    # persian cat category id.
    category_id_2 = 0  # 285

    # Move model and input to device.
    dev = get_device()
    model = model.to(dev)
    x = x.to(dev)

    return model, x, category_id_1, category_id_2


if __name__ == "__main__":
    model, x, category_id, _ = get_example_data()

    # Contrastive excitation backprop.
    saliency = contrastive_excitation_backprop(
        model,
        x,
        category_id,
        saliency_layer='extractor.features.1',
        contrast_layer='extractor.features.4',
        # classifier_layer='classifier.6',
    )

    # Plots.
    plot_example(x, saliency, 'contrastive excitation backprop', category_id)
