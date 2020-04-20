#!"C:\Projects\University\Machine Learning\ml-regression\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'TPOT==0.9.5','console_scripts','tpot'
__requires__ = 'TPOT==0.9.5'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('TPOT==0.9.5', 'console_scripts', 'tpot')()
    )
