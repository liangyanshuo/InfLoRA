from methods.inflorab5 import InfLoRAb5
from methods.inflora_ca import InfLoRA_CA
from methods.inflora_ca1 import InfLoRA_CA1
from methods.inflora import InfLoRA
from methods.inflora_domain import InfLoRA_domain
from methods.inflorab5_domain import InfLoRAb5_domain
from methods.sprompt_coda import SPrompts_coda
from methods.sprompt_l2p import SPrompts_l2p
from methods.sprompt_dual import SPrompts_dual

def get_model(model_name, args):
    name = model_name.lower()
    options = {'sprompts_coda': SPrompts_coda,
               'sprompts_l2p': SPrompts_l2p,
               'sprompts_dual': SPrompts_dual,
               'inflorab5': InfLoRAb5,
               'inflora': InfLoRA,
               'inflora_domain': InfLoRA_domain,
               'inflorab5_domain': InfLoRAb5_domain,
               'inflora_ca': InfLoRA_CA,
               'inflora_ca1': InfLoRA_CA1,
               }
    return options[name](args)

