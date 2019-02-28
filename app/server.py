from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1'
model_file_name = 'model'
classes = ['prunus_spinosa', 'fraxinus_excelsior', 'salix_matsudana', 'celtis_occidentalis', 'syringa_vulgaris', 'sambucus_nigra', 'elaeagnus_angustifolia', 'aesculus_hippocastamon', 'laburnum_anagyroides', 'prunus_padus', 'fagus_sylvatica', 'liriodendron_tulipifera', 'euonymus_europaea', 'gymnocladus_dioicus', 'pyrus_communis', 'staphylea_pinnata', 'corylus_colurna', 'amelanchier_spicata', 'populus_alba', 'acer_pseudoplatanus', 'sorbus_intermedia', 'ailanthus_altissima', 'cercis_canadensis', 'armeniaca_vulgaris', 'viburnum_opulus', 'betula_pendula', 'ulmus_minor', 'salix_caprea', 'ptelea_trifoliata', 'cotinus_coggygria', 'acer_platanoides', 'ginkgo_biloba', 'ilex_aquifolium', 'populus_deltoides', 'ulmus_laevis', 'magnolia_denudata', 'populus_tremula', 'prunus_domestica', 'tilia_cordata', 'acer_campestre', 'corylus_avellana', 'quercus_rubra', 'acer_saccharinum', 'alnus_glutinosa', 'koelreuteria_paniculata', 'sorbus_torminalis', 'persica_vulgaris', 'salix_babylonica', 'quercus_petraea', 'sambucus_racemosa', 'sophora_japonica', 'rhamnus_cathartica', 'viburnum_lantana', 'acer_negundo', 'liquidambar_styraciflua', 'morus_alba', 'gleditsia_triacanthos', 'malus_domestica', 'ficus_carica', 'acer_tataricum', 'carpinus_betulus', 'tilia_tomentosa', 'robinia_pseudo-acacia', 'padus_avium', 'crataegus_sanguinea', 'frangula_alnus', 'acer_palmatum', 'chionanthus_virginicus', 'paulownia_tomentosa', 'catalpa_bignonioides', 'sorbus_aria', 'ulmus_glabra', 'populus_nigra', 'prunus_mahaleb', 'ulmus_pumila', 'maclura_pomifera', 'stewartia_pseudocamellia', 'cerasus_vulgaris', 'juglans_regia', 'sorbus_aucuparia', 'platanus_acerifolia', 'hippophae_rhamnoides', 'quercus_robur']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

