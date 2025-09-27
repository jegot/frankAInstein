# frankAInstein  --working name, we can change it if you want

Description: Visual representation of StableDiffusion's entire image-to-image process including:
  1. Variantional Auto Encoding
  2. Latent space
  3. Diffusion
  4. Decoding

**Audience is intended to be children learning about how AI models generate new images.

Model used: 
  - runwayml/stable-diffusion-v1-5 (diffusion and primary generation)
  - CompVis/stable-diffusion-v1-4 (VAE and reconstructing latent space images)

Kid-friendly story analogy:
  - Characters: Scientist (User), Robot (VAE), Artist (Diffuser).
  - Analogies: Machine (Encoder and Decoder), Door (bottleneck and latent space)
      
    1. The scientist gets a picture he wants to change the style of. He tries to run straight to the artist, but his picture is too big to fit through the door.
    2. Above the door is a sign that says 'latent-space only'
    3. A robot sitting off to the side taps on a machine. The scientist hands the picture to the robot who runs it through the machine.
    4. Out of the machine comes small, blurry version of the picture. The robot hands this to the scientist who can now fit through the door into 'latent space'.
    5. Into the latent space, the scientist meets an artist who takes the new condensed picture. He slowly starts spraying it with spray paint until it is just noise.
    6. The artist turns to the scientist and asks him to select a style from the 'style wall'. 
    7. The scientist chooses his style, and the artist begins removing noise with a paint brush.
    8. Finally, a new picture is created. It is still blurry and small like the other one.
    9. The scientist goes back through the door to the robot, who switches the setting on the machine from 'encode' to 'decode'
    10. The picture goes through the machine in the opposite direct now, and out pops the new picture in the scientist's preferred style.
