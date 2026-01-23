// cypress.config.js
module.exports = {
  e2e: {
    baseUrl: "http://127.0.0.1:5501",
    specPattern: "cypress/e2e/**/*.cy.js",
    supportFile: "cypress/support/commands.js",
    video: false,
    chromeWebSecurity: false,
    defaultCommandTimeout: 15000,
    pageLoadTimeout: 60000,
  },
};
