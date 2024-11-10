from llava.conversation import Conversation
from llava.conversation import *
import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

WKW="wrong_keyword"
WDT="Wrong_details"

WKWN="wrong_keyword_negative"
WDTN="wrong_details_negative"

WKWP="wrong_keyword_positive"
RDTP="replace_details_positive"
import warnings


class ConversationDiffusion:
    """A class that keeps all conversation history."""
    def __init__(self, system: str,
                roles: List[str],
                messages: List[List[str]],
                offset: int,
                sep_style: SeparatorStyle = SeparatorStyle.SINGLE,
                sep: str = "###",
                sep2: str = None,
                version: str = "Unknown",
                skip_next: bool = False,
                process_prompt_mode: str= WDT):
        self.system = system
        self.roles = roles
        self.messages = messages
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
        self.version = version
        self.skip_next = skip_next
        self.process_prompt_mode = process_prompt_mode
    

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def process_prompt(self, prompt):
        if self.process_prompt_mode == WKW:
            processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", which objects or details are lacked or spoiled in the image? Keyword only, no sentence or phrase"
        elif self.process_prompt_mode == WDT:
            processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", Details out the information in the image does not match with the prompt or is not realistic."
        elif self.process_prompt_mode == WKWN:
            # processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", please find out the information in the image does not match with the prompt or is not realistic. Phrases only, no sentence"
            # processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Describe the errors and wrong details in the image with short phrases only, no sentence"
            processed_prompt = f"Given prompt \"${prompt}\" and corresponding generated image {DEFAULT_IMAGE_TOKEN}, please describe the errors and wrong details in the image that does not match with given prompt in short phrases only, no sentence, no reasoning details. For example, blur cat, distorted tables, distorted hands.etc."
        elif self.process_prompt_mode == WDTN:
            processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", Details out the information in the image does not match with the prompt or is not realistic."
        elif self.process_prompt_mode == WKWP:
            processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", which objects or details are lacked or spoiled in the image? Keyword only, no sentence or phrase"
        elif self.process_prompt_mode == RDTP:
            # processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", replace the prompt with exact the meaning of the given prompt with emphasizing more on the details in the original prompt but missed in the image"
            # processed_prompt = f"{DEFAULT_IMAGE_TOKEN} \n Given prompt \"{prompt}\", Adding details to prompt on how to improve the details in the image with short phrases only, no sentence."
            # processed_prompt = f"Given prompt \"{prompt}\" and corresponding generated image {DEFAULT_IMAGE_TOKEN}, use another prompt to improve generated image's details and match with original given prompt"
            # processed_prompt = f"Given prompt \"{prompt}\" and corresponding generated image {DEFAULT_IMAGE_TOKEN}, use another prompt with different words but have the exact meaning with original prompt to improve generated image's details"
            processed_prompt = f"Given prompt \"{prompt}\" and corresponding generated image {DEFAULT_IMAGE_TOKEN}, use a different prompt the exact meaning with original prompt to improve generated image's details with short phrases within 77 words"
        else:
            processed_prompt = prompt
            warnings.warn(f"Current {self.process_prompt_mode} is not supported, the prompt will be kept the same")
            pass
        return processed_prompt
    
    def post_process_prompt(self, prompt):
        if self.process_prompt_mode == WKW:
            post_prompt = f"with enhancing details for {prompt}"
        elif self.process_prompt_mode == WDT:
            post_prompt = prompt
        else:
            warnings.warn(f"Current mode {self.process_prompt_mode} is not supported, keep the same")
            post_prompt = prompt
        return post_prompt
    
    def given_diffusion_prompt(self, prompt):
        processed_prompt = self.process_prompt(prompt)
        self.append_message(self.roles[0], processed_prompt)
        self.append_message(self.roles[1], None)

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return ConversationDiffusion(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_llava_v1_diffusion2 = ConversationDiffusion(
    system="The images have some wrong details compared to the prompt, The assistant help to details out the wrong details in the image",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

    

conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_df": conv_llava_v1_diffusion,
    "llava_v1_dfv2": conv_llava_v1_diffusion2,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}



