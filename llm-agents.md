---
description: 
globs: 
alwaysApply: true
---
# Be agentic about reading the repo

You can browse the repo. Whenever you think there might be relevant code that is not mentioned, search for those files and read the relevant code. Do not reimplement functionality that already exists. Do not make changes incompatible with existing code. Don't ask the user to show you a file you need, search for the file yourself.

---

---
description: 
globs: 
alwaysApply: true
---
- every Python source file (i.e. in the src directory) should start with a comment explaining what that file is for, and listing any major functions in it that might be used elsewhere.
- every Python script file (i.e. in the scripts directory) should start with a comment giving an example of how to run it (remember that it's `python -m scripts.whatever`-style), and an explanation of the args

If you see that a file does not match this, that you are editing, make sure to update it. Also make sure to keep the comments and the file contents synced. 