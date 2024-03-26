from urllib.request import urlretrieve

url = (
    "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
)
filename = "emnist-letters.zip"

urlretrieve(url, filename)
print("done")