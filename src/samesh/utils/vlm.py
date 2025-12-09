from io import BytesIO
import base64
import PIL


from google import genai
from google.genai.types import RawReferenceImage, MaskReferenceImage, MaskReferenceConfig, EditImageConfig, \
    Part, Content, GenerateContentConfig, SafetySetting



def assert_valid_key(key, valid_keys, name):
    """Helper function to validate a key is in valid_keys"""
    if key not in valid_keys:
        raise ValueError(f"Invalid {name}: {key}. Must be one of {list(valid_keys.keys())}")


class Gemini():
    """
    Class for interfacing with supported Gemini models
    """
    # TODO: Is this actually accurate?
    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "9:16": (736, 1408),
        "16:9": (1408, 736),
    }
    IMAGE_SHAPES = {res for res in RESOLUTIONS.values()}

    VERSIONS = {
        "gemini-2.0-flash-preview-image-generation": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 8192,
        },
        "gemini-2.5-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
    }
    def __init__(
        self,
        project,
        location="global",
        model="gemini-2.0-flash-preview-image-generation",
    ):
        """
        Args:
            project (str): Name of the project to use when calling the Gemini client
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
        """
        self.project = project
        print("="*100)
        print(f"USING PROJECT: {self.project}")
        print("="*100)
        self.location = location
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Gemini model")
        self.model = model
        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

    def __call__(
        self,
        prompt,
        image_paths=None,
        images=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_paths (None or str or list of str): If specified, absolute path(s) corresponding to reference image(s)
                to use as part of the overall prompt
            images (None or PIL.Image or list of PIL.Image): If specified, PIL.Image(s) to use as part of the overall prompt
            temperature (float): Temperature of the model when querying. Lower values correspond to more deterministic
                outputs
            top_p (float): Determines the cumulative probability of top-p tokens to select from probabilistically.
                E.g.: If top_p=0.7 and tokens a, b, c have probabilities of 0.4, 0.3, 0.2 respectively, only tokens
                a and b will be sampled from
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of google.genai.types.GenerateContentResponse: Stream of responses generated from Gemini
        """
        parts = [Part.from_text(text=prompt)]
        if image_paths is not None:
            image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
            msg1_images = []
            for image_path in image_paths:
                msg1_images.append(Part.from_bytes(
                    data=self.encode_image_from_path(image_path),
                    mime_type="image/png",
                ))
            parts = msg1_images + parts
        if images is not None:
            images = [images] if isinstance(images, PIL.Image.Image) else images
            msg1_images = []
            for image in images:
                msg1_images.append(Part.from_bytes(
                    data=self.encode_PIL_image(image),
                    mime_type="image/png",
                ))
            parts = msg1_images + parts
        contents = [Content(
            role="user",
            parts=parts,
        )]

        generate_content_config = GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_output_tokens=self.VERSIONS[self.model]["max_tokens"],
            response_modalities=self.VERSIONS[self.model]["modalities"],
            safety_settings=[SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_IMAGE_HATE",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_IMAGE_HARASSMENT",
                threshold="OFF"
            ), SafetySetting(
                category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
                threshold="OFF"
            )],
        )

        result = None
        for i in range(n_retries):
            if result is not None:
                break
            print(f"Querying Gemini [{self.model}]: attempt {i + 1} of {n_retries}...")
            _result = []
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if print_results:
                        print(chunk.text, end="")
                    _result.append(chunk)
                result = _result
            except Exception as e:
                print(f"\nFailed attempt {i + 1} of {n_retries}: {e}")
                print(f"\nFailed attempt {i + 1} of {n_retries}")

        return result

    def get_result_text(self, result):
        return "".join(res.text for res in result)

    def get_result_images(self, result):
        images = []
        for res in result:
            for part in res.candidates[0].content.parts:
                if part.inline_data:
                    # The image data is in base64 encoded format within part.inline_data.data
                    image_data = part.inline_data.data

                    # You can then process this data, for example, save it as an image file
                    # Decode the base64 data and open it with PIL (Pillow)
                    image = PIL.Image.open(BytesIO(image_data))
                    images.append(image)
        return images

    @staticmethod
    def encode_image_from_path(image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            bytes: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def encode_PIL_image(image: PIL.Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str



def check_segmentation_prompt():

    return "You are an expert in image analysis and object segmentation.\n\n" + \
    "### Task Description ###\n\n" + \
    "You will be given a sequence of images of an object. These images are rendered from different viewpoints of the object. \n\n" + \
    "The images alternate between a rendered image of the object and a segmented image of the object. \n\n" + \
    "Each segment is labeled with a number. The color of the segment is the same as the color of the number. \n\n" + \
    "Each color represents a different segment of the object. The colors are consistent throughout the sequence of images. \n\n" + \
    "However, the object may be over segmented. Your task is to identify the parts of the object that are segmented correctly. \n\n" + \
    "All parts of the object that move together and are part of the same articulated component should be labeled with the same number. For example, a handle of a drawer should be labeled with the same number as the drawer. \n\n" + \
    "All parts of the object that do not move together should be labeled with a different number. For example, the body of a drawer should be labeled with a different number than the wheels. This is because wheels can rotate independently of the body of the drawer. \n\n" + \
    "Please group the segments that correspond to the parts of the object that move together and provide this as a list of lists. \n\n" + \
    "Additionally, provide your reasoning for the grouping as a list of strings. \n\n" + \
    "Provide the list as a JSON object with the following fields: \n\n" + \
    f"{{'parts': [[int, int, int], [int, int, int]], 'reasoning': [str, str]}}" + \
    "### Example ###\n\n" + \
    "### Output ###\n\n" + \
    "{{'parts': [[1, 2, 3], [4, 5, 6]], 'reasoning': ['The parts labeled 1 and 2 move together, so they should be labeled with the same number.', 'The parts labeled 3 and 4 do not move together, so they should be labeled with different numbers.']}}" 