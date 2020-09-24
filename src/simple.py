from dataclasses import dataclass, asdict
import json, yaml
import argparse


@dataclass
class TextGenerationConfig:
    model_type: str
    model_name_or_path: str
    cached_dir: str
    max_length: int
    temperature: float
    top_k: int
    top_p: int
    stop_token: str
    do_sample: bool

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value




if __name__ == "__main__":
    # simple_config = TextGenerationConfig.from_json("../config_files/simple_generation_config.json")
    simple_config = TextGenerationConfig.from_yaml("./config/simple_generation_config.yml")
    print("Model name: {}".format(simple_config.model_name_or_path))

    config_dict = asdict(simple_config)
    print("Temperature value: {}".format(config_dict["temperature"]))