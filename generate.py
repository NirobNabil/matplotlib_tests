from DeepImageSearch import Load_Data, Search_Setup
import  DeepImageSearch
print(DeepImageSearch.__file__)

# Load images from a folder
image_list = Load_Data().from_folder(['./images/images2/'])

 # Set up the search engine, You can load 'vit_base_patch16_224_in21k', 'resnet50' etc more then 500+ models 
st = Search_Setup(image_list=image_list, model_name='resnet34', pretrained=True, image_count=None)

# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()

# Add new images to the index
# st.add_images_to_index(['image_path_1', 'image_path_2'])

# Get similar images
# st.get_similar_images(image_path='./images/sphx_glr_axes_box_aspect_003_2_00x.png', number_of_images=10)

# Plot similar images
