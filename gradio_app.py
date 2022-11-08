import gradio as gr
from exploration import SiameseNet
from torchvision import transforms as T
import torch

model = SiameseNet.load_from_checkpoint("checkpoints/best_model-v13.ckpt")
# model = SiameseNet.load_from_checkpoint("checkpoints/best_model-v1.ckpt")
im_trans = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def predict(image1, image2, image3):
    print("Predicting")
    anchor  = im_trans(image1)
    other1  = im_trans(image2)
    other2  = im_trans(image3)
    im1_emb = model(anchor.view(1, *anchor.shape))
    im2_emb = model(other1.view(1, *other1.shape))
    im3_emb = model(other2.view(1, *other2.shape))
    # dist1 = torch.nn.PairwiseDistance()(im1_emb, im2_emb).item()
    # dist2 = torch.nn.PairwiseDistance()(im1_emb, im3_emb).item()
    dist1 = torch.sqrt(torch.sum((im1_emb - im2_emb)**2)).item()
    dist2 = torch.sqrt(torch.sum((im1_emb-im3_emb)**2)).item()
    if dist1 < dist2:
        # return 2.
        return image1, image2, dist1, dist2, abs(dist2 -dist1)
    else:
        return image1, image3, dist1, dist2, abs(dist1 - dist2)
    # dist = torch.nn.CosineSimilarity()(im1_emb, im2_emb)

    # return dist.item()
    

iface = gr.Interface(
    predict,
    [
        gr.inputs.Image(label="Anchor"),
        gr.inputs.Image(),
        gr.inputs.Image(),
    ],
    # gr.outputs.Label(num_top_classes=1),
    [
        gr.outputs.Image(), 
        gr.outputs.Image(), 
        gr.outputs.Label(label="Distance 1", num_top_classes=1),
        gr.outputs.Label(label="Distance 2", num_top_classes=1),
        gr.outputs.Label(label="Distance Delta", num_top_classes=1)
    ],
    title="Siamese Network",
    description="A siamese network that can tell if two images are the same or not.",
    allow_flagging=False,
    examples=[
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg", "https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"],
        ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://www.thesprucepets.com/thmb/ZB88IoSodwKNdt6j8qlx4bI4Ly8=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/boxer-dog-breed-1117944-hero-dfe9f67a59ce4ab19ebd274c06b28ad1.jpg","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://dogtime.com/assets/uploads/2011/01/file_22916_boxer.jpg"],
        # ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://images.ctfassets.net/440y9b545yd9/2L7thrHWc0rVOEoNJhdNOw/68fdd37c4ce5f5de58a4266383aa36a1/Norwegian-Forest-Cat850.jpg"],
        ["https://azure.wgp-cdn.co.uk/app-yourcat/posts/iStock-1031592516.jpg?&width=480&height=480&bgcolor=ffffff&mode=crop&format=webp&webp.quality=40&scale=both","https://www.purina.co.uk/sites/default/files/styles/square_medium_440x440/public/2022-06/Maine-Coon-Cat.jpg?itok=XrHCK4xn", "https://images.ctfassets.net/440y9b545yd9/2L7thrHWc0rVOEoNJhdNOw/68fdd37c4ce5f5de58a4266383aa36a1/Norwegian-Forest-Cat850.jpg"],
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
