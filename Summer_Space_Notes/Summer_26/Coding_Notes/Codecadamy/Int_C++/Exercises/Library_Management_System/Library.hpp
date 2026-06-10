#include <vector>
#include <string>

using namespace std;
extern double libraryBudget; 

// Library Class Def 
class Library {
  public: 
  void addBook(string title, double bookCost);
  bool borrowBook(string title);
  void listBooks();
  Library();
  static int getTotalBorrowedBooks(); 
  void reserveBook(string title) const;

  private:
  vector<string> bookList; 
  static int totalBorrowedBooks; 
  mutable int reservedCount; 
  mutable vector<string> reservedBooks;




};

