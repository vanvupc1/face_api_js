const container=document.querySelector('#container');
const fileinput=document.querySelector('#file-input');

async function loadTrainingData() {
	const labels = ['Fukada Eimi', 'Rina Ishihara', 'Takizawa Laura', 'Yua Mikami']

	const faceDescriptors = []
	for (const label of labels) {
		const descriptors = []
		for (let i = 1; i <= 4; i++) {
			const image = await faceapi.fetchImage(`/face_api_js/data/${label}/${i}.jpeg`)
			const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor()
			descriptors.push(detection.descriptor)
		}
		faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors))
		Toastify({
			text: `Training xong data của ${label}!`
		}).showToast();
	}

	return faceDescriptors
}
let faceMatcher
async function init(){
    
     
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/face_api_js/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/face_api_js/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/face_api_js/models')
    ])
    
    Toastify({
        text: 'Tải xong Model'
    }).showToast();
    const trainingData = await loadTrainingData()
	faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6)

   

}

init()

fileinput.addEventListener('change', async(e) =>{
    const files = fileinput.files;

    const image = await faceapi.bufferToImage(files[0]);
	const canvas = faceapi.createCanvasFromMedia(image);
    container.innerHTML= ''
    container.append(image);
    container.append(canvas);
    const size = {
		width: image.width,
		height: image.height
	}
    faceapi.matchDimensions(canvas, size)

    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
	const resizedDetections = faceapi.resizeResults(detections, size)

    for (const detection of resizedDetections) {
		const drawBox = new faceapi.draw.DrawBox(detection.detection.box, {
			label: faceMatcher.findBestMatch(detection.descriptor).toString()
		})
		drawBox.draw(canvas)
	}
})
