var prepareCmd = `
docker compose build
`
var publishCmd = `
docker compose push
`
var config = require('semantic-release-preconfigured-conventional-commits');
config.plugins.push(
    ["@semantic-release/exec", {
        "prepareCmd": prepareCmd,
        "publishCmd": publishCmd,
    }],
    ["@semantic-release/github", {
        "assets": [
        ]
    }],
    "@semantic-release/git",
)
module.exports = config