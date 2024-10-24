const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());
app.use(express.static('public'));

const API_KEY = 'your_api_key'; // Replace with your API key
const API_URL = 'https://api.example.com/sentiment'; // Replace with the sentiment analysis API URL



app.post('/analyze', async (req, res) => {
    try {
        const response = await axios.post(API_URL, {
            text: req.body.text
        }, {
            headers: {
                'Authorization': `Bearer ${API_KEY}`
            }
        });
        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log('Server is running on port 3000');
});
