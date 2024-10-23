

async function searchKeyword() {
    const keyword = document.getElementById('search-bar').value;

    if (!keyword) {
        alert('Please enter a keyword!');
        return;
    }

    try {
        const response = await fetch('https://capstone-3-omek.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: keyword }),
        });

        const data = await response.json();

        const sentimentOutput = document.getElementById('sentiment-output');
        if (data.error) {
            sentimentOutput.textContent = `Error: ${data.error}`;
        } else if (data.sentiment === undefined) {
            sentimentOutput.textContent = 'Sorry, this is not a climate-related keyword.';
        } else {
            sentimentOutput.textContent = `Sentiment: ${data.sentiment}`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('sentiment-output').textContent = 'An error occurred while processing the request.';
    }
}


async function applyFilters() {
    // Get the selected sentiment filter value
    const sentimentType = document.getElementById('sentiment-type').value;

    // Build the query string with the sentiment filter
    let queryString = '';
    if (sentimentType !== 'all') {
        queryString = `?sentiment_type=${sentimentType}`;
    }

    // Fetch the filtered data from the Flask API
    const response = await fetch(`https://capstone-3-omek.onrender.com/filter-sentiments${queryString}`);
    const data = await response.json();

    // Update the bar chart with the filtered data
    const ctx = document.getElementById('barChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                label: 'Sentiment Distribution',
                data: [data.Positive, data.Negative],
                backgroundColor: ['#36A2EB', '#FF6384']
            }]
        }
    });
}




// Toggle dark mode functionality
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');

    // Store dark mode state in localStorage to retain mode on page reload
    if (document.body.classList.contains('dark-mode')) {
        localStorage.setItem('darkMode', 'enabled');
    } else {
        localStorage.setItem('darkMode', 'disabled');
    }
}

// Check localStorage to apply dark mode on page load if it was enabled previously
function applyDarkModeFromStorage() {
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
    }
}

// Add event listener for the dark mode toggle button
document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('dark-mode-toggle').addEventListener('click', toggleDarkMode);
    applyDarkModeFromStorage();
});


// Example POST request to send a vote
async function submitPoll() {
    const options = document.getElementsByName('poll');
    let selectedOption = null;

    for (let option of options) {
        if (option.checked) {
            selectedOption = option.value;
            break;
        }
    }

    if (selectedOption) {
        // Send vote to the backend
        await fetch('https://capstone-3-omek.onrender.com/vote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ option: selectedOption })
        });

        // Fetch the updated results from the backend
        const response = await fetch('https://capstone-3-omek.onrender.com/results');
        const data = await response.json();
        pollVotes = data.votes;
        totalVotes = pollVotes.reduce((a, b) => a + b, 0);

        displayResults();  // Show the updated chart
    } else {
        alert("Please select an option before submitting.");
    }
}


async function fetchNews() {
    console.log('Fetching news from:', newsUrl); // Log the URL being fetched
    try {
        const response = await fetch(newsUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });

        console.log('Response Status:', response.status); // Log the response status
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Fetched Data:', data); // Log the data fetched
        // ... (rest of your processing logic)
    } catch (error) {
        console.error('Error fetching news:', error);
        newsList.innerHTML = `<p>Error loading news: ${error.message}</p>`;
    }
}

function submitVote(option) {
    fetch('/vote', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ option: option })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Vote counted!');
            fetchResults(); // Refresh results after voting
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error('Error:', error));
}

function fetchResults() {
    fetch('/results')
        .then(response => response.json())
        .then(data => {
            console.log('Poll Results:', data);
            // Update the UI with poll results
        })
        .catch(error => console.error('Error fetching results:', error));
}

function fetchNews() {
    fetch('/news')
        .then(response => response.json())
        .then(data => {
            console.log('News Articles:', data);
            // Update the UI with news articles
        })
        .catch(error => console.error('Error fetching news:', error));
}

