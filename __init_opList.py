def get_opList():
    opDict = [r'https?:\/\/.*\/\w*',
              r'@\w+([-.]\w+)*',
              r'&\w+([-.]\w+)*',
              r'#',
              "["
               u"\U0001F600-\U0001F64F" 
               u"\U0001F300-\U0001F5FF"
               u"\U0001F680-\U0001F6FF" 
               u"\U0001F1E0-\U0001F1FF"
               u"\U00002702-\U000027B0"
               u"\U000024C2-\U0001F251"
               "]+"]
    return opDict
