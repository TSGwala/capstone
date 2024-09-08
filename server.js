const express = require('express');
const bodyParser = require('body-parser');
const bcrypt = require('bcrypt');
const sqlite3 = require('sqlite3').verbose();
const session = require('express-session');
const path = require('path');

const app = express();
const db = new sqlite3.Database('database.db'); // Use the database file

app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Session setup
app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true
}));

// Create a users table if not already present
db.serialize(() => {
    db.run(`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )`);
});

// Middleware to check if user is authenticated
function ensureAuthenticated(req, res, next) {
    if (req.session.userId) {
        return next();
    } else {
        res.redirect('/signin.html');
    }
}

// Root route - Redirect to sign-in page
app.get('/', (req, res) => {
    res.redirect('/signin.html');
});

// Sign-Up Route
app.post('/signup', (req, res) => {
    const { username, email, password } = req.body;

    // Check if the username or email already exists
    db.get(`SELECT * FROM users WHERE email = ? OR username = ?`, [email, username], (err, existingUser) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).send('Internal Server Error');
        }

        if (existingUser) {
            return res.status(400).send('Username or Email already exists');
        }

        // Hash the password and store the new user
        bcrypt.hash(password, 10, (err, hash) => {
            if (err) {
                console.error('Hashing error:', err);
                return res.status(500).send('Internal Server Error');
            }

            db.run(`INSERT INTO users (username, email, password) VALUES (?, ?, ?)`,
                [username, email, hash],
                (err) => {
                    if (err) {
                        console.error('Database error:', err);
                        return res.status(400).send('User already exists or Invalid data');
                    }
                    res.redirect('/signin.html');
                }
            );
        });
    });
});

// Sign-In Route
app.post('/signin', (req, res) => {
    const { username, password } = req.body;

    db.get(`SELECT * FROM users WHERE username = ?`, [username], (err, user) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).send('Internal Server Error');
        }

        if (!user) {
            return res.status(404).send('User not found');
        }

        bcrypt.compare(password, user.password, (err, result) => {
            if (err) {
                console.error('Bcrypt error:', err);
                return res.status(500).send('Internal Server Error');
            }

            if (result) {
                req.session.userId = user.id;
                res.redirect('/index.html');
            } else {
                res.status(401).send('Incorrect Password');
            }
        });
    });
});

// Sign-Out Route
app.get('/signout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            console.error('Session destruction error:', err);
            return res.status(500).send('Error signing out');
        }
        res.redirect('/signin.html'); // Redirect to sign-in page
    });
});

// Serve index.html with authentication check
app.get('/index.html', ensureAuthenticated, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(3000, () => {
    console.log('Server started on http://localhost:3000');
});
