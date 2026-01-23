// cypress/support/commands.js

function seedAuthOnWindow(win, role = {}) {
  win.localStorage.setItem("token", role.token ?? "fake-jwt");
  win.localStorage.setItem("role_level", String(role.role_level ?? 1));
  win.localStorage.setItem("position_id", String(role.position_id ?? 1));
  win.localStorage.setItem("position_name", role.position_name ?? "Admin");
}
Cypress.Commands.add(
  "loginAs",
  (role = {}, path = "/finova/admin/dashboard") => {
    cy.visit(path, {
      onBeforeLoad(win) {
        seedAuthOnWindow(win, role);
      },
    });
  },
);
Cypress.Commands.add("safeClick", (selector) => {
  cy.get("body").then(($body) => {
    if ($body.find(selector).length) {
      cy.get(selector).first().click({ force: true });
    } else {
      cy.log(`safeClick: not found -> ${selector}`);
    }
  });
});
Cypress.Commands.add("assertVisibleIfExists", (selector) => {
  cy.get("body").then(($body) => {
    if ($body.find(selector).length) {
      cy.get(selector).first().should("be.visible");
    } else {
      cy.log(`assertVisibleIfExists: not found -> ${selector}`);
    }
  });
});
Cypress.Commands.add("findLogoutOrUserArea", () => {
  return cy.get("body").then(($body) => {
    const candidates = [
      '[data-action="logout"]',
      "#logout",
      ".logout",
      ".btn-logout",
      ".logout-button",
      'a[href*="logout"]',
      'button:contains("Logout")',
      'button:contains("logout")',
      'a:contains("Logout")',
      'a:contains("logout")',
      ".user",
      ".avatar",
      ".profile",
      '[data-testid*="user"]',
      '[data-testid*="avatar"]',
    ];

    const found = candidates.some((sel) => $body.find(sel).length > 0);

    expect(found, "Logout button OR user menu area should exist").to.eq(true);
  });
});
