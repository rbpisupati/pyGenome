import os
import os.path



__version__ = '0.0.1'
__updated__ = "07.02.2020"
__date__ = "07.02.2020"

log = logging.getLogger(__name__)
def setLog(logDebug):
    log = logging.getLogger()
    if logDebug:
        numeric_level = getattr(logging, "DEBUG", None)
    else:
        numeric_level = getattr(logging, "ERROR", None)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    lch = logging.StreamHandler()
    lch.setLevel(numeric_level)
    lch.setFormatter(log_format)
    log.setLevel(numeric_level)
    log.addHandler(lch)

def die(msg):
    sys.stderr.write('Error: ' + msg + '\n')
    sys.exit(1)

def get_options(program_license,program_version_message):
    inOptions = argparse.ArgumentParser(description=program_license)
    inOptions.add_argument('-V', '--version', action='version', version=program_version_message)
    subparsers = inOptions.add_subparsers(title='subcommands',description='Choose a command to run',help='Following commands are supported')

    return(inOptions)

def main():
    ''' Command line options '''
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = "The main module for pyBsHap"
    program_license = '''%s
    Created by Rahul Pisupati on %s.
    Copyright 2020 Gregor Mendel Institute. All rights reserved.

    Distributed on an "AS IS" basis without warranties
    or conditions of any kind, either express or implied.
    USAGE
    ''' % (program_shortdesc, str(__date__))
    parser = get_options(program_license,program_version_message)
    args = vars(parser.parse_args())
    setLog(args['logDebug'])
    if 'func' not in args:
        parser.print_help()
        return(0)
    try:
        args['func'](args)
        return(0)
    except KeyboardInterrupt:
        return(0)
    except Exception as e:
        logging.exception(e)
        return(2)

if __name__=='__main__':
    sys.exit(main())
