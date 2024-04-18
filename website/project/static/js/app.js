$(document).ready(function () {
  $("nav .on").click(function () {
    $(".alert").css({
      transform: "translateX(0)",
    });
  });

  $("nav .close").click(function () {
    $(".alert").css({
      transform: "translateX(150%)",
    });
  });

  $("nav .out").click(function () {
    $(".alert").css({
      transform: "translateX(150%)",
    });
  });
});

function ToggleSection(element) {
  let button = element.children[0].children[1].classList;
  let section = element.parentElement.classList;
  if (section.contains("open")) {
    section.remove("open");
    button.add("fa-plus");
    button.remove("fa-x");
  } else {
    section.add("open");
    button.remove("fa-plus");
    button.add("fa-x");
  }

  return true;
}

function checkAnswer(element) {
  if (
    element.classList.contains("green") ||
    element.classList.contains("red")
  ) {
    return true;
  }
  let answer = element.parentElement.parentElement.dataset.answer;
  let options = element.parentElement.children;
  let qid = element.parentElement.parentElement.dataset.qid;
  let selection = element.classList[1];
  let status;

  for (let i = 0; i < options.length; i++) {
    options[i].classList.remove("green", "red");
  }

  if (element.children[0].innerText == answer) {
    element.classList.add("green");
    status = "green";
  } else {
    element.classList.add("red");
    status = "red";
  }
  $.ajax({
    url: `/save/${qid}/${status}/${selection}`,
    context: document.body,
  });

  return true;
}

function mark_option(element) {
  if (element.classList.contains("mark")) {
    return true;
  }
  let options = element.parentElement.children;

  for (let i = 0; i < options.length; i++) {
    options[i].classList.remove("mark");
  }

  element.classList.add("mark");

  return true;
}

submit_btn = document.querySelector(".mock-questions .btns .submit-btn");
reset_btn = document.querySelector(".mock-questions .btns .reset-btn");
retest_btn = document.querySelector(".mock-questions .btns .retest-btn");

submit_btn.addEventListener("click", function (e) {
  e.preventDefault();
  let questions = document.querySelectorAll(".mock-questions .question");
  let selection;
  let answer;
  let options;
  let level;
  let score = 0;

  for (let i = 0; i < questions.length; i++) {
    qid = questions[i].dataset.qid;
    options = questions[i].children[1].children;
    for (let j = 0; j < options.length; j++) {
      if (options[j].classList.contains("mark")) {
        options[j].classList.remove("mark");
        selection = options[j].children[0].innerText;
        answer = questions[i].dataset.answer;
        if (selection == answer) {
          options[j].classList.add("green");
          score++;
        } else {
          options[j].classList.add("red");
        }
      }
      if (options[j].children[0].innerText == questions[i].dataset.answer) {
        options[j].classList.add("green");
      }
    }
    submit_btn.style.display = "none";
    reset_btn.style.display = "none";
    retest_btn.classList.remove("d-none");
  }
  // get title and extract the last word
  let title = document.querySelector("title").innerText;
  title = title.split(" ");
  level = title[title.length - 1];

  // get current time
  let date = new Date();
  let time = date.toLocaleTimeString();
  // get current date
  date = date.toDateString();
  date = date.split(" ");
  date = `${date[2]} ${date[1]}, ${date[3]}`;
  time = `${date} ${time}`;

  $.ajax({
    url: `/mock/${level}/${score}/${time}`,
    context: document.body,
  });
});

reset_btn.addEventListener("click", function (e) {
  e.preventDefault();
  let questions = document.querySelectorAll(".mock-questions .question");
  let options;

  for (let i = 0; i < questions.length; i++) {
    options = questions[i].children[1].children;
    for (let j = 0; j < options.length; j++) {
      options[j].classList.remove("green", "red", "mark");
    }
  }
});

function increaseBalance() {
  var currentBalance = parseInt(
    document.querySelector(".balance-amount").textContent
  );
  var newBalance = currentBalance + 500; 
  document.querySelector(".balance-amount").textContent = newBalance;
  $.ajax({
    url: `/account/save_amount/${newBalance}`,
    context: document.body,
  });
}

function decreaseBalance() {
  var currentBalance = parseInt(
    document.querySelector(".balance-amount").textContent
  );
  var newBalance = currentBalance - 500; 
  document.querySelector(".balance-amount").textContent = newBalance;
  $.ajax({
    url: `/account/save_amount/${newBalance}`,
    context: document.body,
  });
}
