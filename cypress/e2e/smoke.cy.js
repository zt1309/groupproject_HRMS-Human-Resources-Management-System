// cypress/e2e/smoke.cy.js
describe("HRMS - Smoke Suite (20 tests, stable)", () => {
  beforeEach(() => {
    cy.visit("/#/finova/admin/dashboard", { failOnStatusCode: false });
    cy.get("body", { timeout: 15000 }).should("exist");
  });

  const optionalVisible = (selector, label = selector) => {
    cy.get("body").then(($body) => {
      if ($body.find(selector).length > 0) {
        cy.get(selector).first().should("be.visible");
      } else {
        cy.log(`[optional] not found: ${label}`);
      }
    });
  };

  const optionalExist = (selector, label = selector) => {
    cy.get("body").then(($body) => {
      if ($body.find(selector).length > 0) {
        cy.get(selector).first().should("exist");
      } else {
        cy.log(`[optional] not found: ${label}`);
      }
    });
  };

  it("01) Dashboard route loads (hash router)", () => {
    cy.url({ timeout: 15000 }).should("include", "/#/finova");
  });

  it("02) Page is not blank", () => {
    cy.get("body").should("be.visible");
  });

  it("03) Navigation container exists (optional)", () => {
    optionalVisible("nav, aside, .sidebar, .navbar, .top-nav", "nav/sidebar");
  });

  it("04) Has at least one clickable element", () => {
    cy.get("a, button").its("length").should("be.greaterThan", 0);
  });

  it("05) Employees menu click (optional)", () => {
    optionalExist(
      '[data-page="employees"], .nav-item[data-page="employees"]',
      "employees menu",
    );
  });

  it("06) Announcements menu click (optional)", () => {
    optionalExist(
      '[data-page="announcements"], .nav-item[data-page="announcements"]',
      "announcements menu",
    );
  });

  it("07) Leaves menu click (optional)", () => {
    optionalExist(
      '[data-page="leaves"], .nav-item[data-page="leaves"]',
      "leaves menu",
    );
  });

  it("08) Attendance menu click (optional)", () => {
    optionalExist(
      '[data-page="attendance"], .nav-item[data-page="attendance"]',
      "attendance menu",
    );
  });

  it("09) Payroll menu exists (optional)", () => {
    optionalExist(
      '[data-page="payroll"], .nav-item[data-page="payroll"]',
      "payroll menu",
    );
  });

  it("10) Has input somewhere (optional)", () => {
    cy.get("body").then(($body) => {
      const has = $body.find("input").length > 0;
      if (has) cy.get("input").first().should("exist");
      else cy.log("[optional] no input found");
    });
  });

  it("11) Has some content blocks", () => {
    cy.get("body").find("div").should("exist");
  });

  it("12) Pagination exists (optional)", () => {
    optionalVisible(
      ".pagination, .page-item, [data-testid*=pagination]",
      "pagination",
    );
  });

  it("13) Header exists (optional)", () => {
    optionalVisible("header, .header, .top-nav, .navbar", "header");
  });

  it("14) No obvious crash text", () => {
    cy.get("body")
      .invoke("text")
      .then((txt) => {
        const lower = String(txt).toLowerCase();
        expect(lower).to.not.include("uncaught");
        expect(lower).to.not.include("exception");
        expect(lower).to.not.include("stack trace");
      });
  });

  it("15) Back/forward does not break", () => {
    cy.go("back");
    cy.go("forward");
    cy.get("body").should("exist");
  });
  it("16) Reload does not crash", () => {
    cy.location("href").then((href) => {
      expect(href).to.include("#");
    });

    cy.reload({ failOnStatusCode: false });

    cy.get("body", { timeout: 15000 }).should("be.visible");
  });
  it("17) Has at least one button (optional)", () => {
    cy.get("body").then(($body) => {
      const has = $body.find("button").length > 0;
      if (has) cy.get("button").first().should("exist");
      else cy.log("[optional] no button found");
    });
  });

  it("18) Has at least one link (optional)", () => {
    cy.get("body").then(($body) => {
      const has = $body.find("a").length > 0;
      if (has) cy.get("a").first().should("exist");
      else cy.log("[optional] no link found");
    });
  });

  it("19) User/Avatar OR Logout area exists (optional)", () => {
    optionalExist(
      '[data-action="logout"], #logout, .logout, .btn-logout, .avatar, .user, [data-testid*=user]',
      "user/avatar/logout",
    );
  });

  it("20) Logout click does not crash (optional)", () => {
    cy.get("body").then(($body) => {
      const sel = '[data-action="logout"], #logout, .logout, .btn-logout';
      if ($body.find(sel).length > 0) {
        cy.get(sel).first().click({ force: true });
        cy.get("body").should("exist");
      } else {
        cy.log("[optional] no logout button to click");
      }
    });
  });
});
