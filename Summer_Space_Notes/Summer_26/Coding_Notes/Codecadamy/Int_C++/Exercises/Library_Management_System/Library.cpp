#include "Library.hpp"
#include "removeElement.hpp"
#include <iostream>

using namespace std;
int Library::totalBorrowedBooks = 0; 


//List Books Function 
 void Library::listBooks(){
  for (auto titles:bookList){
    cout << titles << endl;
  }
}

//add Books Function
void Library::addBook(string title, double bookCost){
if (bookList.size() < 100){
  bookList.push_back(title);
  libraryBudget -= bookCost;

}
else{
  cout << "Library is full!" << endl; 
}
  
}


// borrowBook Function 
bool Library::borrowBook(string title){
  bool found =  false; 
  for (auto book: bookList){
    if (book == title){
      found = true;
      break;
    }
  }
  if (found){
    removeElement(bookList, title);
    totalBorrowedBooks++;          // <-- the one new line
    cout << "Book borrowed!" << endl;
    return true;
  }else {
     cout << "Book not found!" << endl;
     return false;
  }


}

// getTotalBorrowedBooks() Function 
int Library::getTotalBorrowedBooks(){
  return totalBorrowedBooks;
}

//List initalizer way 
Library::Library(): /*class member*/ reservedCount(/*value*/0){}



// reserveBook(title) function
void Library::reserveBook(string title) const{
  bool inLibrary = false; 
  //Does the book exist in the library 
  for (auto book : bookList){
    if (book == title){
      inLibrary = true;
      break;
    }
  }
  //Is is already reserved?
  bool alreadyReserved = false; 
  for (auto book : reservedBooks){
    if (book == title){
        alreadyReserved = true;
        break;

    }
  }
      if (inLibrary && !alreadyReserved) {
      reservedBooks.push_back(title);
      reservedCount++;
      cout << "Book reserved: " << title << endl;
      cout << "Reserved book count: " << reservedCount << endl;
    } else {
      cout << "Book not reserved!" << endl;
    }

}



