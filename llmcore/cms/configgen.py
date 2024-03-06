try:
    import tomllib as toml
except:
    import toml

template_rag_toml = """
[my_new_config]
my_attribute = "This is a new attribute that I need!"
another_new_attribute = "Here is another one.  Add as many as you like."

[sharepoint_config]
sharepoint_site_url = "https://amdcloud.sharepoint.com/sites/SiteName"
sharepoint_folder = "/Shared Documents/SubFolder Name/Another SubFolder if needed"
username = "{{USERNAME}}"
password = "{{PASSWORD}}"

[llm_service_config]
api_proxy_key = "{{API_PROXY_KEY}}"
api_proxy_url = "https://llm-api.amd.com/azure"
llm_deployment_id = "swe-gpt35-turbo-exp1"
vector_db_url = "http://dvue-vctrdb-vm1.amd.com:8080"
vector_db_api_key = false
embedding_engine = "text-embedding-ada-002"

[query_config]
system_role_prompt = "You are an assistant that helps people find information from the IT Knowledge base.\n You will provide answers exclusively from the context provided below.\n The context is based on the IT Knowledge Base Articles.\n You do not possess any knowledge beyond the given context and cannot offer instructions \n for getting, installing or downloading applications unless mentioned in the context.\n Rules:\n - Answer questions using only the context provided. NEVER use information outside of the Context Provided.\n - If the context is empty, respond with 'Sorry, I could not find your query in the IT Knowledge Base. Please check the list of available service requests below or provide more context in your query and try again!'.\n - If you can answer a question without context, state 'Sorry, I could not find your query in the IT Knowledge Base. Please check the list of available service requests below or provide more context in your query and try again!'.\n - Provide detailed instructions and highlight any prerequisites.\n - 'vpn' can be inferred as GlobalProtect VPN.\n - Don't skip but also don't add or modify any links or URLs from the context in your response.\n - Format the response nicely, use bullets if necessary.\n - Never ask the user to refer back to the context.\n - Example: If a question is about an application not mentioned in the context, do not attempt to answer the question.\n - Example: If the question is 'how do I get Outlook' and if we don't have any relevant context that can answer this, repond 'Sorry, I could not find your query in the IT Knowledge Base. Please check the list of available service requests below or provide more context in your query and try again!'\n Context: "
all_top_n = -1
context_top_n = 3
top_n_documents = 3
temperature = 0

[[query_config.vector_classes]]
label = "CustOpsArticle_Class"
class = "CustOpsArticle"
properties = ["text", "filename", "url"]
as_context = true
url_property = "url"

[[query_config.vector_classes]]
label = "Blogs"
class = "WTSXilinxBlogs"
properties = ["text", "url",]
as_context = true
url_property = "url"

[[query_config.vector_classes]]
label = "Community"
class = "WTSXilinxCommunity"
properties = ["text", "linked_community_question", "url"]
as_context = true
url_property = "url"
top_by_certainty = 0.9

[[query_config.vector_classes]]
label = "Knowledge Base"
class = "WTSXilinxKnowledge"
properties = ["text", "url",]
as_context = true
url_property = "url"

[app_server_config]
server_port = 5000

[logging_config]
loggers = ["system_logger", "feedback_logger"]

[logging_config.system_logger]
logger_enabled = false
logger_path = "logs"
logger_level = "DEBUG"

[logging_config.feedback_logger]
logger_enabled = false
logger_path = "logs/feedback"
"""

def generate_config_template(file=None, chatbot_type="rag"):
    chatbot_type_template_map = {
        "rag": template_rag_toml,
        }
    if file is None:
        file = "config.toml-template"
    with open(file, "w") as f:
        if chatbot_type in chatbot_type_template_map:
            f.write(chatbot_type_template_map[chatbot_type])
        else:
            print(f"Unsupported chatbot type: {chatbot_type}. Please select one of ","|".join(chatbot_type_template_map.keys()))


class ChatbotConfig:
    system_logger = None
    feedback_logger = None
    def validate(self):
        pass
    def __str__(self):
        return str(self.__dict__)

class BaseConfig:
    def __init__(self, cfg_dict=None):
        if cfg_dict is not None:
            for attr_name, attr_value in cfg_dict.items():
                setattr(self, attr_name, attr_value)

    def post_init(self):
        pass

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return str(self.__dict__)

class LLMServiceConfig(BaseConfig):
    pass

class LoggingConfig(BaseConfig):
    pass
    
class SharepointConfig(BaseConfig):
    pass

class QueryConfig(BaseConfig):
    vector_classes = []
    def post_init(self):
        self.vector_classes_map = {}
        for vector_class in self.vector_classes:
            self.vector_classes_map[vector_class["label"]] = vector_class

class AppServerConfig(BaseConfig):
    pass

def load_config(config_file: str) -> ChatbotConfig:
    with open(config_file, 'r') as f:
        config_str = f.read()
    return load_config_str(config_str)

def load_config_str(config_str: str) -> ChatbotConfig:
    domain_class_map = {
        'llm_service_config': LLMServiceConfig,
        'logging_config': LoggingConfig,
        'app_server_config': AppServerConfig,
        'query_config': QueryConfig,
        'sharepoint_config': SharepointConfig
        }
    toml_cfg = toml.loads(config_str)
    new_keys = [key for key in toml_cfg.keys() if key not in domain_class_map.keys()]
    for key in new_keys:
        domain_class_map[key] = BaseConfig
    chatbot_config = ChatbotConfig()
    for domain, domain_cfg in toml_cfg.items():
        setattr(chatbot_config, domain, domain_class_map[domain](domain_cfg))
        getattr(chatbot_config, domain).post_init()

    chatbot_config.validate()
    return chatbot_config

def save_config(config: ChatbotConfig, config_file: str):
    save_cfg = {}
    for domain in config.__dict__:
        save_cfg[domain] = getattr(config, domain).__dict__
    with open(config_file, 'w') as f:
        toml.dump(save_cfg, f)