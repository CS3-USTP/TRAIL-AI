import { assertEquals } from "jsr:@std/assert";

const cases = [
    // Joe's custom inputs
    { 
        premise: "is it allowed",                 
        hypothesis: "why not",  
        coherent: true 
    },
    { 
        premise: "why not",                       
        hypothesis: "is it allowed", 
        coherent: true 
    },
    { 
        premise: "can i wear a skirt",            
        hypothesis: "can i wear blue jeans", 
        coherent: false 
    },
    { 
        premise: "can i wear a skirt",            
        hypothesis: "if its long", 
        coherent: true 
    },
    { 
        premise: "can i wear a skirt",            
        hypothesis: "can i access it", 
        coherent: true 
    },
    { 
        premise: "dog is barking it",             
        hypothesis: "animal is noisy", 
        coherent: true 
    },
    { 
        premise: "animal is noisy",               
        hypothesis: "dog is barking it", 
        coherent: true 
    },
    { 
        premise: "can i wear a skirt",            
        hypothesis: "can i hack", 
        coherent: false 
    },
    { 
        premise: "why is the sky blue",           
        hypothesis: "just tell me", 
        coherent: true 
    },
    { 
        premise: "why is the sky blue",           
        hypothesis: "what does that mean", 
        coherent: true 
    },
    { 
        premise: "just tell me",                  
        hypothesis: "why is the sky blue", 
        coherent: false 
    },
    { 
        premise: "can i hack",                    
        hypothesis: "my friend got caught what will they do", 
        coherent: true 
    },
    { 
        premise: "hello",                         
        hypothesis: "what are you", 
        coherent: true 
    },
    { 
        premise: "hello",                         
        hypothesis: "who are you", 
        coherent: true 
    },
    { 
        premise: "what are you",                  
        hypothesis: "what does ustp mean", 
        coherent: true 
    },
    { 
        premise: "what does ustp mean",           
        hypothesis: "why is the sky blue", 
        coherent: false 
    },
    { 
        premise: "she bought a laptop yesterday", 
        hypothesis: "she is an expert in programming", 
        coherent: false 
    },
    { 
        premise: "what is the time",              
        hypothesis: "its barking loudly", 
        coherent: false 
    },
    { 
        premise: "its barking loudly",            
        hypothesis: "what is the time", 
        coherent: false 
    },
    { 
        premise: "what is cs3 who are they",      
        hypothesis: "what is their goal", 
        coherent: true 
    },
    { 
        premise: "what is their goal",            
        hypothesis: "i dont have enough money to continue studying", 
        coherent: false 
    },
    { 
        premise: "a man eats something",          
        hypothesis: "a man is driving down the road", 
        coherent: false 
    },
    // Entailment: One clearly implies the other
    { 
        premise: "Are students required to wear uniforms?",            
        hypothesis: "Is the dress code mandatory?", 
        coherent: true 
    },
    { 
        premise: "Final exams are scheduled in the last week of the semester.",            
        hypothesis: "Students will have exams in the final week.", 
        coherent: true 
    },
    { 
        premise: "Students must maintain a minimum GPA to stay in their program.",            
        hypothesis: "A low GPA can lead to academic probation.", 
        coherent: true 
    },

    // Contradiction: The statements directly oppose each other
    { 
        premise: "The library is open 24/7 for students.",            
        hypothesis: "The library closes at 10 PM.", 
        coherent: false 
    },
    { 
        premise: "Plagiarism results in academic penalties.",            
        hypothesis: "Plagiarism is encouraged for creative projects.", 
        coherent: false 
    },
    { 
        premise: "Students are allowed to bring outside food to the cafeteria.",            
        hypothesis: "Outside food is strictly prohibited in the cafeteria.", 
        coherent: false 
    },

    // Neutral: Unrelated or no clear implication
    { 
        premise: "The student council election will be held next month.",            
        hypothesis: "The university recently introduced new parking rules.", 
        coherent: false 
    },
    { 
        premise: "Midterm exams cover the first half of the semester's lessons.",            
        hypothesis: "The university gym offers yoga classes every Friday.", 
        coherent: false 
    },
    { 
        premise: "Students can request an academic transcript online.",            
        hypothesis: "The campus bookstore sells second-hand textbooks.", 
        coherent: false 
    },
    { 
        premise: "Students can access the library with their university ID.",            
        hypothesis: "The football team won the championship last year.", 
        coherent: false 
    },
    { 
        premise: "Professors must submit final grades within two weeks after exams.",            
        hypothesis: "The cafeteria offers a discount for reusable containers.", 
        coherent: false 
    },
    { 
        premise: "The university provides free counseling services for students.",            
        hypothesis: "Tuition fees are coherent to increase next semester.", 
        coherent: false 
    },
    { 
        premise: "Final-year students must complete a capstone project.",            
        hypothesis: "The university shuttle service runs every 30 minutes.", 
        coherent: false 
    },
    { 
        premise: "Lab equipment must be handled with care and returned after use.",            
        hypothesis: "The university has partnered with a local company for internships.", 
        coherent: false 
    },
    { 
        premise: "Students must maintain a minimum GPA to retain their scholarships.",            
        hypothesis: "The gym is open from 6 AM to 10 PM on weekdays.", 
        coherent: false 
    },
    { 
        premise: "All research proposals must be approved by the ethics committee.",            
        hypothesis: "The university bookstore has a new collection of hoodies.", 
        coherent: false 
    },
    { 
        premise: "Late submission of assignments may result in grade deductions.",            
        hypothesis: "Students can apply for dormitory accommodation online.", 
        coherent: false 
    },
    { 
        premise: "Professors are required to hold at least two office hours per week.",            
        hypothesis: "The computer lab requires a reservation for weekend use.", 
        coherent: false 
    },
    { 
        premise: "Graduation ceremonies are held twice a year, in June and December.",            
        hypothesis: "Students can bring guests to the university cafeteria.", 
        coherent: false 
    },

    // Question-based: One statement poses a question while the other responds
    { 
        premise: "What are the university’s policies on attendance?",            
        hypothesis: "Students must attend at least 80% of their classes.", 
        coherent: true 
    },
    { 
        premise: "Can students request an extension on assignments?",            
        hypothesis: "Professors may grant extensions under special circumstances.", 
        coherent: true 
    },
    { 
        premise: "What is the penalty for missing a final exam?",            
        hypothesis: "Final exam absences require a valid excuse and rescheduling.", 
        coherent: false
    },

    // Follow-ups: The second statement naturally follows from the first
    { 
        premise: "Students must complete all prerequisite courses before enrolling in advanced subjects.",            
        hypothesis: "Can I take Advanced Calculus without passing Basic Calculus?", 
        coherent: false 
    },
    { 
        premise: "The university has a strict no-smoking policy on campus.",            
        hypothesis: "Where are the designated smoking areas?", 
        coherent: false 
    },
    { 
        premise: "Graduating students must settle all outstanding fees.",            
        hypothesis: "Can I receive my diploma if I have unpaid tuition?", 
        coherent: true
    },
    { 
        premise: "What does that mean?",            
        hypothesis: "Can you explain it in simpler terms?", 
        coherent: true 
    },
    { 
        premise: "How does that work?",            
        hypothesis: "Give me an example.", 
        coherent: true 
    },
    { 
        premise: "Is that allowed?",            
        hypothesis: "Can you clarify the rules?", 
        coherent: true 
    },
    { 
        premise: "What happens next?",            
        hypothesis: "Walk me through the process.", 
        coherent: true 
    },
    { 
        premise: "Why is that important?",            
        hypothesis: "Can you elaborate on its significance?", 
        coherent: true 
    },
    // Topic Changes: Shifts to an unrelated topic
    { 
        premise: "Students must submit their theses before the deadline.",            
        hypothesis: "Does the cafeteria serve vegetarian options?", 
        coherent: false 
    },
    { 
        premise: "The university encourages students to join extracurricular activities.",            
        hypothesis: "The computer lab has the latest gaming consoles.", 
        coherent: false 
    },
    { 
        premise: "All first-year students must attend orientation.",            
        hypothesis: "The library has an extensive collection of science fiction books.", 
        coherent: false 
    }
];

for (const test of cases) {
    Deno.test(`Coherence Check: "${test.premise}" → "${test.hypothesis}"`, async () => {
        const response = await fetch("http://localhost:8000/coherence-check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ premise: test.premise, hypothesis: test.hypothesis })
        });

        const data = await response.json();

        assertEquals(
            data.coherence,
            test.coherent,
            `
            premise="${test.premise}", 
            hypothesis="${test.hypothesis}", 
            category=${data.category},
            score=${data.score},
            got=${data.coherence}, 
            coherent=${test.coherent},
            results=${JSON.stringify(data.results, null, 2)}
            `
        );
    });
}
