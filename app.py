import gradio as gr
from model import SiameseNet
from torchvision import transforms as T
import torch
from icecream import ic
import onnxruntime as ort

# model = SiameseNet.load_from_checkpoint("checkpoints/best_model-v9.ckpt")
# model.eval() # TODO why does this fix it?

ort_session = ort.InferenceSession("model.onnx")
im_trans = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def predict(image1, image2):
    print("Predicting")
    ic(image1.shape)
    anchor  = im_trans(image1)
    ic(anchor.shape)
    other1  = im_trans(image2)
    ic(other1.shape)
    anchor = anchor.unsqueeze(0)
    other1 = other1.unsqueeze(0)
    ic(other1.shape)

    ort_inputs = {ort_session.get_inputs()[0].name: anchor.detach().numpy(), ort_session.get_inputs()[1].name: other1.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ic(ort_outs)
    # y_hat = model(anchor, other1)
    # ort_inputs = {model.get_inputs()[0].name: anchor.detach().numpy(), model.get_inputs()[1].name: other1.detach().numpy()}
    # y_hat = model(anchor.view(1, *anchor.shape), other1.view(1, *other1.shape))
    y_hat = torch.tensor(ort_outs[0][0])
    ic(y_hat.shape)
    ic(y_hat)

    return torch.sigmoid(y_hat).item()
    

iface = gr.Interface(
    predict,
    [
        gr.inputs.Image(label="Anchor"),
        gr.inputs.Image(),
    ],
    # gr.outputs.Label(num_top_classes=1),
    [
        gr.outputs.Label(label="Distance 1", num_top_classes=1),
    ],
    title="Siamese Network",
    description="A siamese network that can tell if two images are the same or not.",
    allow_flagging=False,
    examples=[
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg", "https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"],
        ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"],
        ["https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg"],
        ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://images.ctfassets.net/440y9b545yd9/2L7thrHWc0rVOEoNJhdNOw/68fdd37c4ce5f5de58a4266383aa36a1/Norwegian-Forest-Cat850.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://images.ctfassets.net/440y9b545yd9/2L7thrHWc0rVOEoNJhdNOw/68fdd37c4ce5f5de58a4266383aa36a1/Norwegian-Forest-Cat850.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both", "https://images.ctfassets.net/440y9b545yd9/2L7thrHWc0rVOEoNJhdNOw/68fdd37c4ce5f5de58a4266383aa36a1/Norwegian-Forest-Cat850.jpg"],
    ]
    # examples=[
    #     [
    #         ["https://www.oxford-animals.org/wp-content/uploads/2019/06/german-shepherd-dog-1.jpg"],
    #         ["https://www.oxford-animals.org/wp-content/uploads/2019/06/german-shepherd-dog-1.jpg"]
    #     ],
    #     [
    #         ["https://www.oxford-animals.org/wp-content/uploads/2019/06/german-shepherd-dog-1.jpg"],
    #         ["https://www.oxford-animals.org/wp-content/uploads/2019/06/german-shepherd-dog-2.jpg"]
    #     ]
    # ]
)

iface.launch()
