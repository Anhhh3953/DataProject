import random


# PROXIES_LIST = [
#     {
#         "http":"socks4://185.32.4.110:4153",
#         "https":"socks4://185.32.4.110:4153"
#     }
# ]

HTTP_PROXY_LIST = [
    "http://user:password@proxy1.com:8000",
    "http://user:password@proxy2.com:8000",
    "http://user:password@proxy3.com:8000",
]
def get_random_proxy():
    return random.choice(HTTP_PROXY_LIST)
