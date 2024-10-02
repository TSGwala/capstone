async function searchKeyword() {
    const keyword = document.getElementById('search-bar').value;

    if (!keyword) {
        alert('Please enter a keyword!');
        return;
    }

    try {
        
        const response = await fetch('http://127.0.0.1:5000/predict', {
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
    const response = await fetch(`http://localhost:5000/filter-sentiments${queryString}`);
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

// async function fetchTrendingTopics() {
//     const response = await fetch('http://127.0.0.1:5000/trending-topics');
//     const data = await response.json();
//     const trendingTopicsList = document.getElementById('trending-topics-list');
    
//     // Clear previous results
//     trendingTopicsList.innerHTML = '';

//     // Populate the trending topics list
//     if (data.trending_topics) {
//         data.trending_topics.forEach(topic => {
//             const listItem = document.createElement('li');
//             listItem.textContent = topic;
//             trendingTopicsList.appendChild(listItem);
//         });
//     } else {
//         trendingTopicsList.innerHTML = '<li>No trending topics found.</li>';
//     }
// }

// // Fetch trending topics when the page loads
// fetchTrendingTopics();


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
        await fetch('http://localhost:5000/vote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ option: selectedOption })
        });

        // Fetch the updated results from the backend
        const response = await fetch('http://localhost:5000/results');
        const data = await response.json();
        pollVotes = data.votes;
        totalVotes = pollVotes.reduce((a, b) => a + b, 0);

        displayResults();  // Show the updated chart
    } else {
        alert("Please select an option before submitting.");
    }
}


