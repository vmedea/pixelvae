from urllib import request, parse
import json
import websocket

class ComfyUIAPI:
    '''
    Programmatic access to ComfyUI API.
    '''
    def __init__(self, url):
        self.url = url
        self.ws_url = url.replace('http', 'ws') + "ws" # XXX do more intelligently

        self.ws = websocket.WebSocket()
        self.ws.connect(self.ws_url)

        # {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": "ddeeb2dd8f0d4de6a4988131a5a8a285"}}
        status = self._recv()
        assert(status['type'] == 'status')
        self.client_id = status['data']['sid']

    def _recv(self):
        return json.loads(self.ws.recv())

    def execute_prompt(self, prompt):
        '''
        Submit a prompt, return all returned images in an dictionary per node id.
        '''

        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req =  request.Request(self.url + "prompt", data=data)
        with request.urlopen(req) as response:
            #rv = response.read()
            rv = json.load(response)
            prompt_id = rv['prompt_id']

        out = {}
        while True:
            rec = self._recv()
            #{'type': 'executed', 'data': {'node': '5', 'output': {'images': [{'filename': 'ComfyUI_00001_.png', 'subfolder': '', 'type': 'temp'}]}, 'prompt_id': '3a440860-301f-4dd6-97fa-fea12e7acc79'}}
            #{'type': 'executing', 'data': {'node': None, 'prompt_id': '3a440860-301f-4dd6-97fa-fea12e7acc79'}}
            if rec['type'] == 'executed' and rec['data']['prompt_id'] == prompt_id:
                if 'output' in rec['data']:
                    output = rec['data']['output']
                    if 'images' in output:
                        # Fetch any associated images
                        png_data = []
                        for image in output['images']:
                            # [{'filename': 'ComfyUI_00002_.png', 'subfolder': '', 'type': 'temp'}]
                            req = request.Request(self.url + "view?" + parse.urlencode(image))
                            with request.urlopen(req) as response:
                                png_data.append(response.read())
                        out[rec['data']['node']] = png_data
            if rec['type'] == 'executing' and rec['data']['prompt_id'] == prompt_id and rec['data']['node'] is None:
                # End of execution
                break

        return out

    def close(self):
        self.ws.close()

