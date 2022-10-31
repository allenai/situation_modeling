import os
import sys
import traceback
import argparse
from situation_modeling import load_module
from .register import load_external_project

USAGE = "python -m situation_modeling mode [options]"

### library `modes`
MODES = {}
MODES["runner"]                   = "situation_modeling.runner"
MODES["babi_data"]                = "situation_modeling.tools.generate_babi_policies"
MODES["sentence_transformers"]    = "situation_modeling.tools.train_sentence_embeddings"
MODES["cluttr_results"]           = "situation_modeling.tools.cluttr_results"
MODES["cluttr"]                   = "situation_modeling.tools.cluttr_results"

def main(argv):
    """Main execution point 

    :param argv: the cli input to situation_modeling module 
    """
    if not argv:
        exit('Please specify mode and settings! Current modes="%s" exiting...' % '; '.join(MODES))
    
    mode = MODES.get(argv[0],None)
    if mode is None:
        exit('Unknown mode=%s, please choose from `%s`' % (argv[0],'; '.join(MODES)))
 
    ## load external modules
    if '--external_project' in argv:
        try: 
            project_path = argv[argv.index('--external_project')+1]
            ## load module
            load_external_project(project_path)

        except IndexError:
            exit('Must provide valid external project path!')

    ## try to execute the target module
    try:
        mod = load_module(mode)
        mod.main(argv)
    except Exception as e:
        print("Uncaught error encountered during execution!",file=sys.stderr)
        traceback.print_exc(file=sys.stdout)
        raise e
    finally:
        ## close the stdout
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
        if sys.stderr != sys.__stderr__:
            sys.stderr.close()

if __name__ == "__main__":
    main(sys.argv[1:])
