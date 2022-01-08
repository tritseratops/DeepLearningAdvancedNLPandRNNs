from subprocess import Popen, PIPE
import os

# test_script = os.sep.join([constants.workspace, "scripts", "launchEnv.bat"])
test_script = "launchEnv.bat"

# test_script = "my.bat"

# for line in stdout: #.decode("utf-8").lower()
#     print('line:',line)
#     # print('type(line):', type(line))
#     if "error" in line: # e.decode("utf-8").lower()
#         assert False


# import subprocess
# import tempfile
#
# with tempfile.TemporaryFile() as tempf:
#     proc = subprocess.Popen(test_script, stdout=tempf)
#     proc.wait()
#     tempf.seek(0)
#     print(tempf.read())

# p = Popen(test_script, shell=True, stdout=PIPE).communicate()[0]
# stdout, stderr = p.communicate()
# print('p:',p)

# lines = p.stdout
# err_lines = p.stderr
# print("****")
# print("err_lines:", err_lines)
# print(lines)
# for line in p: #.decode("utf-8").lower()
#     print('line:',line)
#     # print('type(line):', type(line))
#     if "error" in line: # e.decode("utf-8").lower()
#         assert False


# stdout, stderr = p.communicate()

# lines = [x.decode('utf8').strip() for x in stdout.readlines()]
#
# for line in lines:
#     print(line)
# str = stdout.decode("utf-8")
#
# print('str:', str)
# print('stdout:',stdout)
# print('stderr:',stderr)

# find error in stdout

# assert