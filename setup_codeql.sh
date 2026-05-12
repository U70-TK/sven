wget https://github.com/github/codeql-cli-binaries/releases/download/v2.11.1/codeql-linux64.zip
python3 -c "
import zipfile, os, stat
with zipfile.ZipFile('codeql-linux64.zip') as zf:
    for info in zf.infolist():
        zf.extract(info, '.')
        perm = (info.external_attr >> 16) & 0xFFFF
        if perm:
            os.chmod(info.filename, perm)
"
git clone --depth=1 --branch codeql-cli-2.11.1 https://github.com/github/codeql.git codeql/codeql-repo
codeql/codeql pack download --dir=codeql/packages codeql-cpp@0.7.1 codeql-python@0.6.2 codeql/ssa@0.0.16 codeql/tutorial@0.0.9 codeql/regex@0.0.12 codeql/util@0.0.9
cp data_eval/trained/cwe-190/1-c/ArithmeticTainted.ql codeql/codeql-repo/cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql
